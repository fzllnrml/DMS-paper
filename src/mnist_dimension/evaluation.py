"""Comparable support metrics with exhaustive blocked float64 CPU distance search."""
from pathlib import Path
import hashlib,math
import torch
from src.mnist_experiments.common import fingerprint,save_pt,digest
from src.mnist_experiments.data import ReferenceData
from src.mnist_experiments.evaluator import FrozenEvaluator
from src.mnist_experiments.evaluation import semantic_metrics,support_percentiles
from .representation import CachedPCASpace


class KNN:
    def __init__(self,reference,k=5):
        self.reference=reference.detach().cpu().double().flatten(1).contiguous()
        self.norms=self.reference.square().sum(1)
        self.k=k
        if not 1<=k<=len(reference): raise ValueError('Invalid reference count/k')
    def query(self,query):
        values=[];indices=[]
        for q in query.detach().cpu().double().flatten(1).split(64):
            best=torch.full((len(q),self.k),float('inf'),dtype=torch.float64)
            ids=torch.zeros(len(q),self.k,dtype=torch.long)
            qn=q.square().sum(1)
            for begin in range(0,len(self.reference),4096):
                r=self.reference[begin:begin+4096]
                d2=(qn[:,None]+self.norms[begin:begin+len(r)]-2*q@r.T).clamp_min(0)
                # Avoid subtractive cancellation for coincident/near-copy pairs.
                near=(d2<1e-8).nonzero()
                if len(near):d2[near[:,0],near[:,1]]=(q[near[:,0]]-r[near[:,1]]).square().sum(1)
                best,where=torch.cat((best,d2),1).topk(self.k,largest=False)
                ids=torch.cat((ids,torch.arange(begin,begin+len(r)).expand(len(q),-1)),1).gather(1,where)
            # Direct final distances; no approximate index or candidate sampling.
            exact=(q[:,None,:]-self.reference[ids]).square().sum(2).sqrt()
            exact,order=exact.sort(1);ids=ids.gather(1,order)
            values.append(exact.float());indices.append(ids)
        return torch.cat(values),torch.cat(indices)


class StudyEvaluator:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['root'])/'evaluation_cache'
        source=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=True)
        self.data=ReferenceData(source,cfg['data_root'],cfg['download'],False)
        self.train=self.data.images('train',cfg.get('train_limit'))
        count=cfg.get('validation_limit',cfg['reference_validation_count'])
        self.validation=self.data.images('validation',count)
        self.evaluator=FrozenEvaluator.load(cfg['evaluator_checkpoint'],cfg['device'],cfg['batch_size'])
        for key in ('train','validation'):
            if not torch.equal(self.evaluator.metadata[key+'_indices'],self.data.indices[key]):
                raise ValueError('Evaluator split differs from study source')
        self.k=min(cfg['knn_k'],len(self.train))
        pixels_sha=hashlib.sha256(self.train.numpy().tobytes()+self.validation.numpy().tobytes()).hexdigest()
        self.key=fingerprint(dict(version=1,pixels=pixels_sha,evaluator=digest(cfg['evaluator_checkpoint']),
                                  k=self.k,torch=str(torch.__version__),device=str(self.evaluator.device)))
        path=self.root/(self.key+'.pt');self.pixel=KNN(self.train,self.k)
        if path.exists():
            cache=torch.load(path,map_location='cpu',weights_only=True)
            self.train_features=cache['train_features'];self.references=cache['references']
        else:
            self.train_features=self.evaluator.features(self.train)
            valfeatures=self.evaluator.features(self.validation)
            pixel_d,_=self.pixel.query(self.validation)
            feature_d,_=KNN(self.train_features,self.k).query(valfeatures)
            self.references=dict(pixel=pixel_d.mean(1)/28,feature=feature_d.mean(1))
            save_pt(path,dict(train_features=self.train_features,references=self.references))
        self.feature=KNN(self.train_features,self.k)
        self.representation=None;self.reference_rep=None;self.rep_scale=None
    def set_space(self,space,rep_sha):
        if space.name=='pixel':
            self.representation=self.pixel;self.reference_rep=self.references['pixel']
            self.rep_scale=float(self.train.square().mean().sqrt());return
        path=self.root/(fingerprint(dict(base=self.key,representation=rep_sha,d=space.dim))+'.pt')
        train_z=space.encode(self.train)
        self.representation=KNN(train_z,self.k)
        self.rep_scale=float(train_z.square().mean().sqrt())
        if path.exists():self.reference_rep=torch.load(path,map_location='cpu',weights_only=True)['reference']
        else:
            values,_=self.representation.query(space.encode(self.validation))
            self.reference_rep=values.mean(1)/math.sqrt(space.dim)
            save_pt(path,dict(reference=self.reference_rep))
    def evaluate(self,z,space):
        z=z.detach().cpu();images=space.decode(z).cpu()
        if not torch.isfinite(images).all():raise FloatingPointError('Nonfinite decoded candidates')
        logits,features=self.evaluator.evaluate(images);sem=semantic_metrics(logits)
        pixel,indices=self.pixel.query(images);feature,_=self.feature.query(features)
        rep,_=self.representation.query(z)
        pixel= pixel.mean(1)/28;feature=feature.mean(1);rep=rep.mean(1)/math.sqrt(space.dim)
        overshoot=(images.abs()-1).clamp_min(0)
        per=dict(pixel_knn_distance=pixel,feature_knn_distance=feature,representation_knn_rms=rep,
                 representation_knn_relative_scale=rep/max(self.rep_scale,1e-12),
                 pixel_support_percentile=torch.as_tensor(support_percentiles(pixel,self.references['pixel'])['support_percentile']),
                 feature_support_percentile=torch.as_tensor(support_percentiles(feature,self.references['feature'])['support_percentile']),
                 representation_support_percentile=torch.as_tensor(support_percentiles(rep,self.reference_rep)['support_percentile']))
        flat={k:float(v.mean()) for k,v in per.items()}
        flat.update(classifier_confidence=sem['mean_confidence'],represented_class_count=sem['class_coverage'],class_entropy=sem['class_entropy'],
                    decoded_fraction_outside=float((overshoot>0).float().mean()),decoded_max_overshoot=float(overshoot.max()),
                    decoded_mean_overshoot=float(overshoot.mean()))
        return dict(metrics=flat,per_candidate=per,semantics=sem,nearest_train_indices=indices[:,0],
                    reference_key=self.key,representation_scale=self.rep_scale)
