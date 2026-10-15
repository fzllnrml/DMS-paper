"""One paired suite with immutable checkpoints and restartable method outputs."""
from copy import deepcopy
import hashlib,json,resource,shutil,sys,time
from pathlib import Path
import torch,yaml
from src.mnist import MNISTDiffusion
from .common import CountedModel,PixelSpace,gaussian_noise,seed_all,save_pt,write_json,digest,timestamp,git_revision,synchronize,fingerprint,code_digest,merge_dict,plain
from .config import DEFAULT
from .data import ReferenceData
from .evaluator import FrozenEvaluator
from .evaluation import SupportEvaluator,local_stability,directional_curvature
from .search import run_search,baseline_config,resolve_levels
from .pca import PCADiffusion,PCASpace
from . import plots

def flat_metrics(m):
    out={}
    if m.get('semantics'):
        for k in ('mean_confidence','class_coverage','class_entropy'): out[k]=m['semantics'][k]
    for kind in ('pixel','feature'):
        if kind in m:
            out[kind+'_nearest_mean']=float(m[kind]['nearest_distance'].mean()); out[kind+'_knn_mean']=float(m[kind]['knn_mean_distance'].mean())
            out[kind+'_support_percentile']=float(m[kind]['knn_percentiles']['support_percentile'].mean())
    return out

def load_model(path,device):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('format')=='mnist_diffusion_v1': return MNISTDiffusion.from_checkpoint(path,device),PixelSpace(),state
    if state.get('format')=='mnist_pca_diffusion_v1':
        model=PCADiffusion.from_checkpoint(path,device); return model,PCASpace(state['pca'],model.dim),state
    raise ValueError('Unsupported checkpoint format')

def immutable_copy(source,folder):
    sha=digest(source); target=Path(folder)/f'{sha}.pt'; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=sha: raise ValueError('Corrupt immutable checkpoint copy')
    else:
        tmp=target.with_suffix('.tmp'); shutil.copy2(source,tmp); tmp.replace(target)
    return target

def run_suite(config,output,resume=False):
    publication = config.get('kind') == 'publication_search'
    from .publication import configure, definitions, MATCH_TARGETS, STABILITY_METHODS, matched_budget
    cfg = configure(config) if publication else merge_dict(DEFAULT,config)
    output=Path(output); cfg['search'].pop('resolved_level_count',None)
    if cfg['initial_noise_seed'] is None: cfg['initial_noise_seed']=cfg['search_seed']
    methods=cfg['methods']
    if not methods or len(set(methods))!=len(methods): raise ValueError('Empty or duplicate methods')
    if not publication and any(m.startswith('matched_') for m in methods) and 'original' not in methods: raise ValueError('Matched baselines require original in same suite')
    methods=sorted(methods,key=lambda m:m in MATCH_TARGETS) if publication else sorted(methods,key=lambda m:m!='original')
    if cfg['population_size']<1: raise ValueError('population_size must be positive')
    if output.exists() and any(output.iterdir()) and not resume: raise FileExistsError('Run exists; use --resume')
    checkpoint_sha=digest(cfg['checkpoint']); ecfg=cfg['evaluation']; ep=ecfg['evaluator_checkpoint']; esha=digest(ep) if ep else None; implementation=code_digest()
    signature=fingerprint(dict(config=cfg,checkpoint_sha=checkpoint_sha,evaluator_sha=esha,implementation=implementation))
    if (output/'metadata.json').exists() and json.loads((output/'metadata.json').read_text())['signature']!=signature: raise ValueError('Resume config/checkpoint/code changed')
    if resume and (output/'complete.json').exists():
        if json.loads((output/'complete.json').read_text())['signature']!=signature: raise ValueError('Signature mismatch')
        for method in methods:
            for artifact in ('complete.json','summary.json','metrics.json','arrays.pt'):
                if not (output/method/artifact).exists(): raise ValueError('Completed artifact missing')
        return json.loads((output/'summary.json').read_text())
    model,space,state=load_model(cfg['checkpoint'],cfg['device']); actual_seed=state.get('config',{}).get('seed')
    if actual_seed is not None and actual_seed!=cfg['model_seed']: raise ValueError('Configured model_seed disagrees with checkpoint')
    levels,level_meta=resolve_levels(cfg['search']['schedule'],model.schedule)
    if not 1<=cfg['ddim_steps']<=model.schedule.T: raise ValueError('DDIM steps exceed trained schedule')
    if space.name=='pca' and cfg['search']['projection']!='none': raise ValueError('PCA needs projection=none')
    seed_all(cfg['search_seed']); evaluator=FrozenEvaluator.load(ep,cfg['device'],cfg['batch_size']) if ep else None
    if ecfg['require_evaluator'] and evaluator is None: raise ValueError('Train the independent evaluator first')
    if cfg['search']['merge']['metric'].startswith('feature_') and evaluator is None: raise ValueError('Feature merge requires evaluator')
    support=None
    if ecfg['enabled'] or evaluator:
        data=ReferenceData(state,ecfg['data_root'],ecfg['download'],ecfg['include_test'])
        if evaluator:
            for key in ('train','validation'):
                if key+'_indices' not in evaluator.metadata or not torch.equal(evaluator.metadata[key+'_indices'],data.indices[key]): raise ValueError('Evaluator and diffusion data split differ')
        if ecfg['enabled']:
            if publication:
                from .publication_evaluation import PublicationSupport
                support=PublicationSupport(data,evaluator,ecfg,esha)
            else:
                support=SupportEvaluator(data,evaluator,ecfg['knn_k'],ecfg['train_limit'],ecfg['validation_limit'],ecfg['include_test'])
    output.mkdir(parents=True,exist_ok=True); cp=immutable_copy(cfg['checkpoint'],output.parent/'checkpoints'); evcopy=immutable_copy(ep,output.parent/'checkpoints') if ep else None
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    write_json(output/'metadata.json',dict(signature=signature,implementation_sha256=implementation,checkpoint_sha256=checkpoint_sha,checkpoint_copy=str(cp),evaluator_sha256=esha,evaluator_copy=str(evcopy) if evcopy else None,evaluator_validation_accuracy=evaluator.metadata.get('validation_accuracy') if evaluator else None,model_seed=actual_seed,search_seed=cfg['search_seed'],initial_noise_seed=cfg['initial_noise_seed'],git=git_revision(),torch_version=str(torch.__version__),device=str(model.device),created_at=timestamp(),time_schedule=level_meta,space=space.name,interpretation='Mode candidates. Labels are external semantics, not true modes.'))
    if publication:
        import importlib.metadata
        metadata=json.loads((output/'metadata.json').read_text())
        metadata.update(protocol=cfg['protocol'],definition_hash=cfg['definition_hash'],
                        method_definitions={m:definitions()[m] for m in methods},
                        publication_series=cfg['publication_series'],
                        package_versions={p:importlib.metadata.version(p) for p in ('torch','torchvision','numpy','scipy','scikit-learn','matplotlib','PyYAML')},
                        python_version=sys.version,
                        matched_targets={m:MATCH_TARGETS[m] for m in methods if m in MATCH_TARGETS},
                        matched_semantics='Actual target method NFE includes its own DDIM initialization. Final-only merge adds no diffusion evaluations.')
        write_json(output/'metadata.json',metadata)
    condition=deepcopy(cfg)
    for key in ('model_seed','search_seed','initial_noise_seed','checkpoint','device','suite_id'): condition.pop(key,None)
    condition['model_architecture']=model.config; condition_id=fingerprint(condition)
    variance=float(space.state['explained_variance_ratio'][:space.dim].sum()) if space.name=='pca' else None
    reconstruction=float((space.decode(space.encode(support.validation))-support.validation).square().mean().sqrt()) if space.name=='pca' and support else None
    rows=[]; budget=None; budgets={}; evaluation_cache={}
    def evaluate_images(images, initial_count):
        if support is None: return {}
        if not publication: return support.evaluate(images, initial_count)
        key=(hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),initial_count)
        if key not in evaluation_cache: evaluation_cache[key]=support.evaluate(images, initial_count)
        return evaluation_cache[key]
    visual_enabled = cfg['visualization']['enabled'] and (not publication or cfg['search_seed'] in cfg['visualization'].get('search_seeds',[0]))
    for method in methods:
        folder=output/method; completed=folder/'complete.json'
        if resume and completed.exists():
            old=json.loads(completed.read_text())
            if old['signature']!=signature: raise ValueError('Method signature mismatch')
            for name in ('arrays.pt','metrics.json','summary.json'):
                if not (folder/name).exists(): raise ValueError('Method artifact missing')
            row=old['summary']; rows.append(row)
            budgets[method]=row['nfe']
            if method=='original': budget=row['nfe']
            continue
        folder.mkdir(parents=True,exist_ok=True); counted=CountedModel(model,cfg['batch_size'])
        if evaluator: evaluator.images_evaluated=0; evaluator.forward_calls=0
        matched=method.startswith('matched_')
        if publication and matched: budget=budgets[MATCH_TARGETS[method]]
        n=budget//cfg['ddim_steps'] if matched else cfg['population_size']
        if n<1 or matched and n>cfg['max_matched_samples']: raise ValueError('Matched sample count exceeds allowed budget/guard')
        noise=gaussian_noise(n,model.dim,cfg['initial_noise_seed']); synchronize(model.device); started=time.perf_counter()
        if model.device.type=='cuda': torch.cuda.reset_peak_memory_stats(model.device)
        uses_ddim=cfg['initialization']=='ddim' or method.startswith('ddim') or method=='final_refine' or matched
        if uses_ddim:
            if publication and matched:
                # Keep the P-prefix batch boundaries identical to DDIM-P.
                boundary=cfg['population_size']
                initial=counted.ddim(noise[:boundary],cfg['ddim_steps'])
                if n>boundary: initial=torch.cat((initial,counted.ddim(noise[boundary:],cfg['ddim_steps'])))
            else: initial=counted.ddim(noise,cfg['ddim_steps'])
        elif cfg['initialization']=='uniform' and space.name=='pixel': initial=(2*torch.rand(n,model.dim,generator=torch.Generator().manual_seed(cfg['initial_noise_seed']))-1).to(model.device)
        else: raise ValueError('Use ddim initialization, or uniform for pixel space only')
        init_nfe=counted.nfe; search_cfg=deepcopy(cfg['search']); search_cfg['resolved_level_count']=len(levels); search_cfg=definitions()[method] if publication else baseline_config(search_cfg,method)
        found=run_search(counted,initial,search_cfg,space,evaluator); candidates=found['candidates']; synchronize(model.device)
        search_seconds=time.perf_counter()-started; feature_work=evaluator.images_evaluated if evaluator else 0
        budgets[method]=counted.nfe
        if method=='original': budget=counted.nfe
        ev_start=time.perf_counter(); initial_images=space.decode(initial).cpu(); images=space.decode(candidates).cpu()
        imetrics=evaluate_images(initial_images,len(initial)); metrics=evaluate_images(images,len(initial))
        diagnostic=CountedModel(model,cfg['batch_size']); stability=None; curvature=None
        if cfg['stability']['enabled'] and (not publication or method in STABILITY_METHODS): stability=local_stability(diagnostic,candidates,space,dict(cfg['stability'],seed=cfg['search_seed']+1001),evaluator)
        if cfg['curvature']['enabled']: curvature=directional_curvature(diagnostic,candidates,dict(cfg['curvature'],seed=cfg['search_seed']+2001))
        synchronize(model.device)
        anchors=torch.stack([initial[ids].mean(0) for ids in found['lineage']]); displacement=float((candidates-anchors).square().mean(1).sqrt().mean())
        steps=[r for stage in found['stages'] for r in stage['movement']]
        row=dict(suite_id=cfg.get('suite_id',output.name),condition_id=condition_id,method=method,variant=cfg.get('variant','original_geometry'),space=space.name,dimension=model.dim,model_seed=cfg['model_seed'],checkpoint_sha256=checkpoint_sha,search_seed=cfg['search_seed'],initial_noise_seed=cfg['initial_noise_seed'],population_size=cfg['population_size'],initialization='ddim' if uses_ddim else cfg['initialization'],initial_count=len(initial),candidate_count=len(candidates),compression_ratio=len(initial)/len(candidates),nfe=counted.nfe,initialization_nfe=init_nfe,movement_nfe=counted.nfe-init_nfe,diagnostic_nfe=diagnostic.nfe,total_diffusion_nfe=counted.nfe+diagnostic.nfe,forward_calls=counted.forward_calls,feature_geometry_evaluator_images=feature_work,evaluator_images=evaluator.images_evaluated if evaluator else 0,matched_budget=budget if matched else None,unused_matched_budget=budget-counted.nfe if matched else None,search_wall_seconds=search_seconds,evaluation_wall_seconds=time.perf_counter()-ev_start,displacement_rms=displacement,movement_efficiency=displacement/max(1,counted.nfe-init_nfe),clipping_fraction_mean=sum(r['clipping']['fraction_outside_range'] for r in steps)/max(1,len(steps)),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device) if model.device.type=='cuda' else None,mps_allocated_bytes_snapshot=torch.mps.current_allocated_memory() if model.device.type=='mps' else None,pca_variance_retained=variance,validation_reconstruction_rms=reconstruction,merge_metric=search_cfg['merge']['metric'],step_mode=search_cfg['step_mode'],projection=search_cfg['projection'],uses_external_geometry=search_cfg['merge']['metric'].startswith('feature_'),**flat_metrics(metrics))
        if publication:
            row.update(protocol=cfg['protocol'], definition_hash=cfg['definition_hash'],
                       publication_series=cfg['publication_series'], checkpoint=cfg['checkpoint'],
                       evaluation_definition_hash=fingerprint(dict(evaluation={k:v for k,v in ecfg.items() if k not in {'reference_cache','evaluator_checkpoint','data_root'}},stability=cfg['stability'],curvature=cfg['curvature'])),
                       evaluator_sha256=esha, paired_suite_id=cfg.get('suite_id',output.name),
                       initial_gaussian_prefix_sha256=hashlib.sha256(noise[:cfg['population_size']].contiguous().numpy().tobytes()).hexdigest(),
                       initial_ddim_prefix_sha256=hashlib.sha256(initial_images[:cfg['population_size']].contiguous().numpy().tobytes()).hexdigest(),
                       initial_candidate_count=len(initial), final_candidate_count=len(candidates),
                       pixel_knn_distance=row.get('pixel_knn_mean'), feature_knn_distance=row.get('feature_knn_mean'),
                       classifier_confidence=row.get('mean_confidence'), represented_class_count=row.get('class_coverage'),
                       clipping_fraction=row['clipping_fraction_mean'], wall_time=search_seconds,
                       resolved_method_definition=search_cfg)
            if matched:
                row.update(matched_budget(budget,cfg['ddim_steps']), matched_target_method=MATCH_TARGETS[method])
                if row['matched_ddim_nfe'] != counted.nfe: raise RuntimeError('Actual matched NFE differs from accounted NFE')
            if 'memorization' in metrics:
                row.update(metrics['memorization']['summary'])
        if stability: row['stability_return_rate']=float(torch.stack([r['return_rate'] for r in stability['results']]).mean())
        if curvature: row['curvature_fraction_negative']=float(curvature['fraction_negative'].mean())
        save_pt(folder/'arrays.pt',dict(initial_noise=noise,initial_noise_used=uses_ddim,initial=initial.cpu(),candidates=candidates.cpu(),decoded_initial=initial_images,decoded_candidates=images,stages=found['stages'],trajectory=found['trajectory'],lineage=found['lineage'],stability=stability,curvature=curvature))
        stable_json=deepcopy(stability)
        if stable_json:
            for record in stable_json['results']: record.pop('endpoints',None)
        write_json(folder/'metrics.json',dict(initial=imetrics,final=metrics,stability=stable_json,curvature=curvature)); write_json(folder/'summary.json',row)
        with (folder/'levels.jsonl').open('w') as stream:
            for stage in found['stages']: stream.write(json.dumps(plain(stage),allow_nan=False)+'\n')
        if visual_enabled:
            plots.grid(initial_images,folder/'initial.png',cfg['visualization']['max_images']); plots.grid(images,folder/'candidates.png',cfg['visualization']['max_images'])
            plots.trajectory(initial.cpu(),found['trajectory'],space,folder/'trajectory.png',cfg['visualization']['trajectory_particles'])
            if support: plots.neighborhood(images,metrics,support,folder/'neighbors_train_test.png')
            if cfg['visualization']['merge_comparison']:
                for record,stage in zip(found['trajectory'],found['stages']):
                    if stage['merge']['merge_radius'] is not None:
                        plots.merge_comparison(record,dict(search_cfg['merge'],comparison_radius=stage['merge']['merge_radius']),space,evaluator,folder/'merge_mean_medoid.png'); break
        write_json(completed,dict(signature=signature,finished_at=timestamp(),summary=row)); rows.append(row)
    write_json(output/'summary.json',rows)
    if visual_enabled and not publication: plots.summary_plots(rows,output/'plots')
    if visual_enabled and publication:
        from .publication_plots import paired_grid
        paired_grid(output,methods,cfg['visualization']['max_images'])
    write_json(output/'complete.json',dict(signature=signature,finished_at=timestamp(),methods=methods)); return rows
