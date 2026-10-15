"""External semantics, empirical support and local score diagnostics."""
import math
import numpy as np
import torch
from .search import move

def semantic_metrics(logits):
    confidence,predicted=logits.softmax(1).max(1); counts=torch.bincount(predicted,minlength=10); p=counts.float()/len(logits); nz=p[p>0]
    return dict(predicted_class=predicted,classifier_confidence=confidence,class_histogram=counts,class_coverage=int((counts>0).sum()),class_entropy=float(-(nz*nz.log()).sum()),mean_confidence=float(confidence.mean()))

def nearest_neighbors(query,reference,k=5,query_batch=64,reference_batch=2048):
    q=query.detach().cpu().float().flatten(1); r=reference.detach().cpu().float().flatten(1)
    if not 1<=k<=len(r): raise ValueError('Invalid k')
    out=[]; indices=[]
    for batch in q.split(query_batch):
        best=torch.full((len(batch),k),float('inf')); ids=torch.full((len(batch),k),-1,dtype=torch.long)
        for begin in range(0,len(r),reference_batch):
            block=r[begin:begin+reference_batch]
            # Direct subtraction kernel avoids cancellation for near-copies.
            d=torch.cdist(batch,block,compute_mode='donot_use_mm_for_euclid_dist')
            values,where=torch.cat((best,d),1).topk(k,largest=False,dim=1)
            ids=torch.cat((ids,torch.arange(begin,begin+len(block)).expand(len(batch),-1)),1).gather(1,where); best=values
        out.append(best); indices.append(ids)
    return torch.cat(out),torch.cat(indices)

def support_percentiles(values,reference):
    ref=np.sort(np.asarray(reference,dtype=float)); values=np.asarray(values,dtype=float)
    p=50*(np.searchsorted(ref,values,'left')+np.searchsorted(ref,values,'right'))/len(ref)
    return dict(distance_percentile=p,support_percentile=100-p)

class SupportEvaluator:
    def __init__(self,data,evaluator=None,k=5,train_limit=None,validation_limit=256,include_test=False):
        self.data=data; self.evaluator=evaluator; self.k=k; self.train=data.images('train',train_limit); self.validation=data.images('validation',validation_limit)
        self.test=data.images('test') if include_test else None; self.train_features=evaluator.features(self.train) if evaluator else None; self.reference={}
        for name,query,ref in [('pixel',self.validation,self.train),('feature',evaluator.features(self.validation) if evaluator else None,self.train_features)]:
            if query is None: continue
            d,_=nearest_neighbors(query,ref,k)
            if name=='pixel': d=d/math.sqrt(784)
            self.reference[name]=dict(nearest=d[:,0],knn=d.mean(1))
        self.metadata=dict(train_reference_count=len(self.train),validation_reference_count=len(self.validation),train_is_subset=train_limit is not None,validation_is_subset=validation_limit is not None,k=k,pixel_distance_units='RMS normalized pixels',feature_distance_units='L2 frozen features',test_memorization_audit=include_test,test_used_for_tuning=False)
    def evaluate(self,images,initial_count):
        images=images.detach().cpu(); result=dict(number_of_candidates=len(images),compression_ratio=initial_count/len(images),reference_metadata=self.metadata,reference_distributions=self.reference,semantics=None)
        features=None
        if self.evaluator:
            logits,features=self.evaluator.evaluate(images); result['semantics']=semantic_metrics(logits)
        for name,query,ref in [('pixel',images,self.train),('feature',features,self.train_features)]:
            if query is None: continue
            d,idx=nearest_neighbors(query,ref,self.k)
            if name=='pixel': d=d/math.sqrt(784)
            result[name]=dict(nearest_distance=d[:,0],knn_mean_distance=d.mean(1),neighbor_distances=d,train_indices=self.data.indices['train'][idx],reference_positions=idx,nearest_percentiles=support_percentiles(d[:,0],self.reference[name]['nearest']),knn_percentiles=support_percentiles(d.mean(1),self.reference[name]['knn']))
        if self.test is not None:
            d,idx=nearest_neighbors(images,self.test,1); result['test_nearest']=dict(distance=d[:,0]/math.sqrt(784),test_indices=idx[:,0])
        return result

def local_stability(model,candidates,space,cfg,evaluator=None):
    n,d=candidates.shape; rng=torch.Generator().manual_seed(cfg['seed'])
    v=torch.randn(n,cfg['directions'],d,generator=rng); v=(v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)).to(candidates.device)
    before=model.nfe; center,_=move(model,candidates.clone(),cfg['t'],cfg['steps'],cfg['alpha'],cfg['step_mode'],cfg['projection'],space)
    results=[]
    for scale in cfg['scales']:
        perturbed=(candidates[:,None,:]+scale*math.sqrt(d)*v).reshape(-1,d)
        end,_=move(model,perturbed,cfg['t'],cfg['steps'],cfg['alpha'],cfg['step_mode'],cfg['projection'],space); end=end.reshape(n,cfg['directions'],d)
        dist=(end-candidates[:,None,:]).square().mean(-1).sqrt(); centered=(end-center[:,None,:]).square().mean(-1).sqrt(); consistency=None
        if evaluator:
            labels=evaluator.evaluate(space.decode(candidates))[0].argmax(1)
            predicted=evaluator.evaluate(space.decode(end.flatten(0,1)))[0].argmax(1).reshape(n,-1)
            consistency=(predicted==labels[:,None]).float().mean(1)
        results.append(dict(perturbation_rms=scale,endpoint_distance=dist.cpu(),distance_to_unperturbed_endpoint=centered.cpu(),return_rate=(dist<=cfg['return_radius']).float().mean(1).cpu(),semantic_consistency=consistency,endpoints=end.cpu()))
    return dict(results=results,nfe=model.nfe-before,unperturbed_drift=(center-candidates).square().mean(1).sqrt().cpu(),return_radius_rms=cfg['return_radius'],t=cfg['t'])

def directional_curvature(model,candidates,cfg):
    if cfg['epsilon']<=0 or cfg['directions']<1: raise ValueError('Invalid finite difference settings')
    n,d=candidates.shape; v=torch.randn(n,cfg['directions'],d,generator=torch.Generator().manual_seed(cfg['seed']))
    v=(v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)).to(candidates.device); center=candidates[:,None,:].expand_as(v); before=model.nfe
    plus=model.score((center+cfg['epsilon']*v).reshape(-1,d),cfg['t']).reshape_as(v)
    minus=model.score((center-cfg['epsilon']*v).reshape(-1,d),cfg['t']).reshape_as(v)
    values=(((plus-minus)/(2*cfg['epsilon']))*v).sum(-1).cpu(); a=values.numpy()
    return dict(values=values,fraction_negative=(values<0).float().mean(1),mean=values.mean(1),median=np.median(a,axis=1),quantiles=np.quantile(a,[.05,.25,.5,.75,.95],axis=1).T,nfe=model.nfe-before,t=cfg['t'],epsilon=cfg['epsilon'],projection_during_difference=False,interpretation='Directional learned-score diagnostic, not a Hessian certificate')
