"""Same recipe across dimensions, explicit diagnostics and epoch-exact resume."""
from pathlib import Path
import json,time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from src.mnist_experiments.pca import PCADiffusion
from src.mnist_experiments.common import save_pt,write_json,seed_all,digest,fingerprint,synchronize
from .artifacts import signature,start,complete


def recipe(cfg):
    source=torch.load(cfg['recipe_checkpoint'],map_location='cpu',weights_only=True)
    settings=source['config']['training'].copy()
    if source['config']['seed']!=42 or source['epoch']<settings['epochs']:
        raise ValueError('A completed seed42 training checkpoint is required for the common recipe')
    for key in ('epochs','batch_size','lr','weight_decay','grad_clip'):
        if key not in settings: raise ValueError(f'Seed42 recipe lacks {key}')
    if cfg['smoke']: settings['epochs']=cfg['epochs_override']
    model_config=source['model_config']
    if (model_config['T'],model_config['beta_start'],model_config['beta_end'])!=(1000,.0001,.02):
        raise ValueError('Study expects the fixed publication diffusion schedule')
    return settings,source


def train_model(cfg,model_spec,resume=False):
    root=Path(cfg['root']);out=root/'models'/model_spec['id'];basis=root/'representation/basis.pt';latent=root/'representation/latents.pt'
    sig=signature(dict(config=cfg,model=model_spec),[basis,latent,cfg['recipe_checkpoint']])
    if start(out,sig,resume): return out/'best.pt'
    settings,source=recipe(cfg)
    state=torch.load(basis,map_location='cpu',weights_only=True)
    if not cfg['smoke']:
        for key in ('train','validation'):
            if not torch.equal(state[key+'_indices'],torch.as_tensor(source['split_indices'][key])):
                raise ValueError('Representation and seed42 recipe splits disagree')
    data=torch.load(latent,map_location='cpu',weights_only=True);d=model_spec['dimension'];seed=model_spec['model_seed']
    train=data['train'][:,:d].contiguous();validation=data['validation'][:,:d].contiguous()
    seed_all(seed);model=PCADiffusion(dim=d,hidden=cfg['hidden'],device=cfg['device'])
    optimizer=torch.optim.AdamW(model.net.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    order=torch.Generator().manual_seed(seed+1);noise_rng=torch.Generator().manual_seed(seed+2)
    begin,best,history=0,float('inf'),[]
    if resume and (out/'latest.pt').exists():
        old=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
        if old['signature']!=sig: raise ValueError('Training checkpoint signature changed')
        model.net.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        order.set_state(old['order_rng']);noise_rng.set_state(old['noise_rng'])
        begin,best,history=old['epoch'],old['best_loss'],old['history']
    loader=DataLoader(TensorDataset(train),batch_size=settings['batch_size'],shuffle=True,generator=order,num_workers=0)
    write_json(out/'resolved_config.json',dict(config=cfg,model=model_spec,training=settings,
        model_config=model.config,parameter_count=sum(p.numel() for p in model.net.parameters()),
        objective='epsilon MSE averaged per coordinate',validation_noise_seed=314159,
        optimizer='AdamW',checkpoint_selection='minimum full validation epsilon MSE',
        source_recipe_sha256=digest(cfg['recipe_checkpoint']),representation_sha256=digest(basis)))
    def loss(x,generator):
        t=torch.randint(1000,(len(x),),generator=generator).to(model.device)
        # Same full-dimensional random noise prefixes across d for fixed validation.
        noise=torch.randn(len(x),784,generator=generator)[:,:d].to(model.device)
        a=model.schedule.alphas_cumprod[t,None];x=x.to(model.device)
        result=F.mse_loss(model.net(a.sqrt()*x+(1-a).sqrt()*noise,t),noise)
        if not torch.isfinite(result): raise FloatingPointError('Nonfinite diffusion loss')
        return result
    for epoch in range(begin+1,settings['epochs']+1):
        synchronize(model.device);started=time.perf_counter();model.net.train();total=0.
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True);value=loss(batch,noise_rng);value.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(),settings['grad_clip'],error_if_nonfinite=True)
            optimizer.step();total+=value.item()*len(batch)
        model.net.eval();val_rng=torch.Generator().manual_seed(314159);val=0.
        with torch.no_grad():
            for batch in validation.split(settings['batch_size']):val+=loss(batch,val_rng).item()*len(batch)
        val/=len(validation);improved=val<best;best=min(best,val);synchronize(model.device)
        history.append(dict(epoch=epoch,train_loss=total/len(train),validation_loss=val,
                            epoch_seconds=time.perf_counter()-started,train_objects=len(train),validation_objects=len(validation)))
        checkpoint=dict(format='mnist_dimension_diffusion_v1',signature=sig,model=model.net.state_dict(),model_config=model.config,
            model_seed=seed,optimizer=optimizer.state_dict(),order_rng=order.get_state(),noise_rng=noise_rng.get_state(),
            epoch=epoch,best_loss=best,history=history,settings=settings,pca=state,representation_sha256=digest(basis))
        # Best before latest: interruption cannot leave latest advanced past a missing best.
        if improved: save_pt(out/'best.pt',checkpoint)
        save_pt(out/'latest.pt',checkpoint);write_json(out/'history.json',history)
        print(f'{model_spec["id"]} epoch {epoch}/{settings["epochs"]}: train={total/len(train):.6g} val={val:.6g}',flush=True)
    complete(out,sig,['best.pt','latest.pt','history.json','resolved_config.json'])
    return out/'best.pt'
