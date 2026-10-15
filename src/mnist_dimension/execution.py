"""Model-quality checks and paired searches, using the existing movement/sampler."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,time
import torch
from src.mnist import MNISTDiffusion
from src.mnist_experiments.pca import PCADiffusion
from src.mnist_experiments.common import CountedModel,PixelSpace,gaussian_noise,save_pt,write_json,digest,synchronize
from src.mnist_experiments.search import run_search
from src.mnist_experiments.publication import matched_budget,definition_hash
from src.mnist_experiments.run import immutable_copy
from src.mnist_experiments.plots import grid
from .representation import CachedPCASpace,conditional_ci
from .design import METHODS,method_config
from .artifacts import signature,start,complete


def load_model(spec,cfg):
    state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
    if spec['representation']=='pixel':
        model=MNISTDiffusion.from_checkpoint(spec['checkpoint'],cfg['device']);space=PixelSpace()
        if state['config']['seed']!=spec['model_seed']:raise ValueError('Pixel checkpoint seed mismatch')
    else:
        if state.get('format')!='mnist_dimension_diffusion_v1':raise ValueError('Use dimension-study checkpoints, not exploratory PCA checkpoints')
        if state['model_seed']!=spec['model_seed'] or state['model_config']['dim']!=spec['dimension']:
            raise ValueError('PCA checkpoint identity mismatch')
        basis=Path(cfg['root'])/'representation/basis.pt'
        if digest(basis)!=state['representation_sha256']:raise ValueError('Checkpoint basis changed')
        model=PCADiffusion(**state['model_config'],device=cfg['device']);model.net.load_state_dict(state['model'])
        space=CachedPCASpace(state['pca'],spec['dimension'])
    if (model.config['T'],model.config['beta_start'],model.config['beta_end'])!=(1000,.0001,.02):raise ValueError('Unexpected schedule')
    if spec['representation']=='pca' and model.config['hidden']!=cfg['hidden']:raise ValueError('Unexpected dimension model capacity')
    model.net.eval()
    return model,space,state


def paired_noise(n,d,seed):
    return gaussian_noise(n,784,seed)[:,:d].contiguous()


def initialize(counted,n,p,seed):
    noise=paired_noise(n,counted.dim,seed)
    # Preserve the P-prefix batch shape of all paired methods, including matched.
    images=counted.ddim(noise[:min(n,p)],50)
    if n>p:images=torch.cat((images,counted.ddim(noise[p:],50)))
    if not torch.isfinite(images).all():raise FloatingPointError('Nonfinite DDIM samples')
    return noise,images


def sha_tensor(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def validation_by_time(model,space,images,timesteps,batch_size):
    result=[];nfe=0
    x=space.encode(images)
    for t in timesteps:
        noise=paired_noise(len(x),model.dim,314159+t)
        errors=[];prefix_errors=[]
        for i in range(0,len(x),batch_size):
            data=x[i:i+batch_size].to(model.device)
            eps=noise[i:i+len(data)].to(model.device)
            a=model.schedule.alphas_cumprod[t]
            xt=a.sqrt()*data+(1-a).sqrt()*eps
            ts=torch.full((len(data),),t,device=model.device,dtype=torch.long)
            prediction=model.net(xt.reshape(-1,1,28,28) if space.name=='pixel' else xt,ts).flatten(1)
            mse=(prediction-eps).square().mean(1)
            if not torch.isfinite(mse).all():raise FloatingPointError('Nonfinite validation diagnostic')
            errors.append(mse.cpu());nfe+=len(data)
            if space.name=='pca':prefix_errors.append((prediction[:,:min(16,model.dim)]-eps[:,:min(16,model.dim)]).square().mean(1).cpu())
        result.append(dict(t=t,epsilon_mse=float(torch.cat(errors).mean()),per_image_mse=torch.cat(errors).tolist(),prefix16_mse=float(torch.cat(prefix_errors).mean()) if prefix_errors else None,n=len(x)))
    return result,nfe


def quality_check(cfg,spec,evaluator,resume=False):
    out=Path(cfg['root'])/'quality'/spec['id']
    deps=[spec['checkpoint'],cfg['evaluator_checkpoint'],cfg['source_checkpoint']]
    if spec['representation']=='pca':deps.append(str(Path(cfg['root'])/'representation/basis.pt'))
    latest_path=Path(spec['checkpoint']).with_name('latest.pt')
    if latest_path.exists():deps.append(str(latest_path))
    sig=signature(dict(config=cfg,model=spec),deps)
    if start(out,sig,resume):return json.loads((out/'quality.json').read_text())
    model,space,state=load_model(spec,cfg)
    rep_sha=state.get('representation_sha256','identity_pixels')
    for key in ('train','validation'):
        saved=state['split_indices'][key] if space.name=='pixel' else state['pca'][key+'_indices']
        expected=evaluator.data.indices[key]
        if space.name=='pca':expected=expected[:cfg.get(key+'_limit')]
        if not torch.equal(torch.as_tensor(saved),expected):raise ValueError('Model/reference splits disagree')
    evaluator.set_space(space,rep_sha)
    val=evaluator.data.images('validation',cfg.get('validation_limit',cfg['quality_validation_count']))[:cfg['quality_validation_count']]
    valrows,val_nfe=validation_by_time(model,space,val,cfg['quality_timesteps'],cfg['batch_size'])
    counted=CountedModel(model,cfg['batch_size']);noise,z=initialize(counted,cfg['sanity_count'],cfg['sanity_count'],cfg['sanity_seed'])
    metrics=evaluator.evaluate(z,space)
    reconstructed=evaluator.evaluate(space.encode(val),space)
    if spec['representation']=='pca':
        repquality=next(r for r in json.loads((Path(cfg['root'])/'representation/quality.json').read_text()) if r['dimension']==spec['dimension'])
    else:
        repquality=dict(retained_variance=1.,validation_reconstruction_rms=0.,train_coordinate_rms=float(evaluator.train.square().mean().sqrt()),information_loss_warning=False,reconstruction_rms_ci95=[0.,0.])
    history=state.get('history',[])
    latest=Path(spec['checkpoint']).with_name('latest.pt')
    if latest.exists():
        latest_state=torch.load(latest,map_location='cpu',weights_only=True);history=latest_state.get('history',history)
    training_loss=state.get('best_loss',state.get('best_val_loss',state.get('best_validation_loss')))
    flags=[]
    if repquality['information_loss_warning']:flags.append('retained_variance_below_0.90')
    if metrics['metrics']['classifier_confidence']<.5:flags.append('sanity_confidence_below_0.5')
    if metrics['metrics']['decoded_fraction_outside']>.25:flags.append('sanity_outside_fraction_above_0.25')
    if not history:flags.append('training_history_unavailable_in_checkpoint')
    if len(history)>1 and history[-1]['validation_loss']>history[0]['validation_loss']:flags.append('final_validation_loss_above_initial')
    result=dict(**spec,parameter_count=sum(p.numel() for p in model.net.parameters()),model_config=model.config,
        checkpoint_sha256=digest(spec['checkpoint']),representation_sha256=rep_sha,
        publication_definition_hash=definition_hash(),representation_quality=repquality,
        best_training_validation_loss=training_loss,validation_by_time=valrows,
        sanity_feature_knn_ci95=conditional_ci(metrics['per_candidate']['feature_knn_distance'].numpy()),
        diagnostic_epsilon_mse=sum(r['epsilon_mse'] for r in valrows)/len(valrows),
        diagnostic_prefix16_mse=(sum(r['prefix16_mse'] for r in valrows)/len(valrows) if space.name=='pca' else None),
        diagnostic_epsilon_mse_ci95=conditional_ci(torch.tensor([r['per_image_mse'] for r in valrows]).mean(0).numpy()),
        quality_diagnostic_nfe=val_nfe+counted.nfe,sanity_nfe=counted.nfe,sanity=metrics['metrics'],
        reconstructed_real= reconstructed['metrics'],history=history,flags=flags,
        interpretation='Flags are diagnostic annotations. Finite poor-quality models are not excluded.')
    grid(space.decode(z).cpu(),out/'sanity.png',cfg['max_visual_images'])
    grid(torch.stack((val,space.decode(space.encode(val))),1).reshape(-1,1,28,28),out/'reconstruction.png',cfg['max_visual_images']*2)
    save_pt(out/'arrays.pt',dict(noise=noise,sanity=z.cpu(),evaluation=metrics,reconstructed_evaluation=reconstructed))
    write_json(out/'quality.json',result);write_json(out/'resolved_config.json',cfg)
    immutable_copy(spec['checkpoint'],Path(cfg['root'])/'checkpoints')
    complete(out,sig,['quality.json','arrays.pt','sanity.png','reconstruction.png','resolved_config.json'])
    return result


def movement_summary(found,initial,space,mode):
    stages=found['stages'];records=[r for s in stages for r in s['movement']]
    regular=[r for s in stages if s['kind']=='level' for r in s['movement']]
    final=[r for s in stages if s['kind']=='final' for r in s['movement']]
    def average(rs,key):return float(torch.stack([r[key] for r in rs]).mean()) if rs else 0.
    factor=math.sqrt(initial.shape[1]/784)
    proposed=average(regular,'proposed_step_rms');actual=average(regular,'step_rms')
    net=(found['candidates']-initial).square().mean(1).sqrt()
    path=torch.stack([r['step_rms'] for r in records]).sum(0) if records else torch.zeros(len(initial))
    expected=.03/(math.sqrt(initial.shape[1]) if mode=='l2' else 1)
    return dict(regular_proposed_step_rms=proposed,regular_actual_step_rms=actual,
        final_actual_step_rms=average(final,'step_rms'),expected_unclipped_regular_step_rms=expected if records else 0.,
        actual_to_expected_step_ratio=actual/expected if records else 0.,
        regular_actual_pixel_step_rms=actual*factor,
        displacement_rms=float(net.mean()),displacement_l2=float(net.mean())*math.sqrt(initial.shape[1]),
        displacement_pixel_rms=float(net.mean())*factor,cumulative_rms_path=float(path.mean()),
        cumulative_pixel_rms_path=float(path.mean())*factor,
        boundary_fraction_mean=sum(r['clipping']['fraction_outside_range'] for r in records)/max(1,len(records)),
        boundary_max_overshoot=max([r['clipping']['max_overshoot'] for r in records],default=0.),
        clipping_fraction_mean=(sum(r['clipping']['fraction_outside_range'] for r in records)/max(1,len(records)) if space.name=='pixel' else 0.))


def search_job(cfg,job,evaluator,resume=False):
    out=Path(job['output']);quality_path=Path(cfg['root'])/'quality'/f'{job["representation"]}_d{job["dimension"]}_m{job["model_seed"]}'/'quality.json'
    sig=signature(dict(config=cfg,job=job),[job['checkpoint'],quality_path,cfg['evaluator_checkpoint']])
    if start(out,sig,resume):
        from .artifacts import validate_complete
        if not all(validate_complete(out/m) for m in METHODS):raise ValueError('Completed suite has corrupt/missing method artifacts')
        return json.loads((out/'summary.json').read_text())
    model,space,state=load_model(job,cfg);quality=json.loads(quality_path.read_text())
    evaluator.set_space(space,quality['representation_sha256'])
    p=cfg['population_size'];rows=[];budget=None;memo={}
    write_json(out/'resolved_config.json',dict(config=cfg,job=job,method_definitions={m:method_config(m,space.name) for m in METHODS}))
    for method in METHODS:
        folder=out/method;msig=signature(dict(suite=sig,method=method),[])
        if start(folder,msig,resume):
            row=json.loads((folder/'summary.json').read_text());rows.append(row)
            if method=='rms_v3f2':budget=row['nfe']
            continue
        n=budget//50 if method=='matched_ddim' else p
        counted=CountedModel(model,cfg['batch_size']);synchronize(model.device);started=time.perf_counter()
        noise,initial=initialize(counted,n,p,job['initial_noise_seed'])
        search=method_config(method,space.name);found=run_search(counted,initial,search,space)
        if not torch.isfinite(found['candidates']).all():raise FloatingPointError('Search produced nonfinite candidates')
        synchronize(model.device);wall=time.perf_counter()-started
        if method=='rms_v3f2':budget=counted.nfe
        key=sha_tensor(found['candidates'])
        if key not in memo:memo[key]=evaluator.evaluate(found['candidates'],space)
        metrics=memo[key];movement=movement_summary(found,initial,space,search['step_mode'])
        row=dict(protocol=cfg['protocol'],suite_id=job['id'],representation=space.name,dimension=job['dimension'],model_seed=job['model_seed'],
                 search_seed=job['search_seed'],initial_noise_seed=job['initial_noise_seed'],method=method,population_size=p,
                 nfe=counted.nfe,wall_time=wall,diagnostic_nfe=0,initial_count=n,final_count=len(found['candidates']),
                 checkpoint=job['checkpoint'],checkpoint_sha256=digest(job['checkpoint']),representation_sha256=quality['representation_sha256'],
                 model_config=model.config,publication_definition_hash=definition_hash(),evaluation_key=evaluator.key,
                 initial_noise_prefix_sha256=sha_tensor(noise[:p]),initial_ddim_prefix_sha256=sha_tensor(initial[:p]),
                 parameter_count=quality['parameter_count'],retained_variance=quality['representation_quality']['retained_variance'],
                 validation_reconstruction_rms=quality['representation_quality']['validation_reconstruction_rms'],
                 diagnostic_epsilon_mse=quality['diagnostic_epsilon_mse'],best_training_validation_loss=quality['best_training_validation_loss'],
                 model_quality_flags=quality['flags'],**metrics['metrics'],**movement)
        row['displacement_relative_to_rep_scale']=row['displacement_rms']/max(evaluator.rep_scale,1e-12)
        if method=='matched_ddim':
            row.update(matched_budget(budget));row['matched_target_method']='rms_v3f2'
            if row['nfe']!=row['matched_ddim_nfe']:raise RuntimeError('Matched NFE mismatch')
        flags=list(quality['flags'])
        if row['decoded_fraction_outside']>.25:flags.append('candidate_outside_fraction_above_0.25')
        if row['clipping_fraction_mean']>.25:flags.append('search_clipping_above_0.25')
        row['flags']=flags
        save_pt(folder/'arrays.pt',dict(noise=noise,initial=initial.cpu(),candidates=found['candidates'].cpu(),stages=found['stages'],evaluation=metrics))
        write_json(folder/'summary.json',row)
        if job['search_seed'] in cfg['visual_search_seeds']:
            grid(space.decode(found['candidates']).cpu(),folder/'candidates.png',cfg['max_visual_images'])
        complete(folder,msig,['arrays.pt','summary.json']);rows.append(row)
    write_json(out/'summary.json',rows)
    complete(out,sig,['summary.json','resolved_config.json'])
    return rows
