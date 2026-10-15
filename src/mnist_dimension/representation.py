"""One train-only, unwhitened orthonormal PCA basis for all nested dimensions."""
from pathlib import Path
import hashlib
import torch
import numpy as np
from src.mnist_experiments.data import ReferenceData
from src.mnist_experiments.pca import PCASpace
from src.mnist_experiments.common import save_pt,write_json,digest
from .artifacts import signature,start,complete


class CachedPCASpace(PCASpace):
    def __init__(self,state,dim):
        super().__init__(state,dim)
        self._buffers={}
    def buffers(self,device):
        key=str(device)
        if key not in self._buffers:
            self._buffers[key]=(self.components.to(device),self.mean.to(device))
        return self._buffers[key]
    def encode(self,x):
        components,mean=self.buffers(x.device)
        return (x.flatten(1)-mean)@components.T
    def decode(self,z):
        components,mean=self.buffers(z.device)
        return (z@components+mean).reshape(-1,1,28,28)


def conditional_ci(values,root=False):
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(123)
    estimates=np.array([rng.choice(a,len(a),replace=True).mean() for _ in range(1000)])
    if root:estimates=np.sqrt(estimates)
    return np.quantile(estimates,[.025,.975]).tolist()


def fit_basis(images,dimensions):
    """Double precision streamed covariance: avoids full 55k-by-784 SVD workspace."""
    n=len(images)
    if n<2 or not dimensions or any(d<1 or d>784 for d in dimensions):
        raise ValueError('Invalid fit dimensions/sample count')
    mean=torch.zeros(784,dtype=torch.float64)
    for raw in images.split(2048): mean+=raw.double().flatten(1).div(127.5).sub(1).sum(0)
    mean/=n
    cov=torch.zeros(784,784,dtype=torch.float64)
    for raw in images.split(2048):
        x=raw.double().flatten(1).div(127.5).sub(1)-mean
        cov+=x.T@x
    values,vectors=torch.linalg.eigh(cov/(n-1))
    order=torch.arange(783,-1,-1)
    values=values[order].clamp_min(0)
    if values.sum()<=0:raise ValueError('Zero-variance training representation')
    components=vectors[:,order].T
    # Resolve sign ambiguity. Degenerate eigenspaces remain library-dependent;
    # exact saved basis and hashes, not refitting, define all later runs.
    sign=torch.sign(components[torch.arange(784),components.abs().argmax(1)])
    components*=sign[:,None]
    return dict(format='mnist_dimension_representation_v1',mean=mean.float(),components=components.float(),
                explained_variance=values.float(),explained_variance_ratio=(values/values.sum()).float(),
                dimensions=list(dimensions),fit_split='train',whiten=False,normalization='uint8 / 127.5 - 1',
                fit_n=n,rank_estimate=int((values>values.max()*1e-10).sum()),solver='streamed float64 covariance eigh',
                geometry='orthonormal unwhitened coordinates, no per-dimension scale fitting')


def prepare(cfg,resume=False):
    out=Path(cfg['root'])/'representation'
    sig=signature(cfg,[cfg['source_checkpoint']])
    if start(out,sig,resume): return out/'basis.pt'
    source=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=True)
    data=ReferenceData(source,cfg['data_root'],cfg['download'],True)
    train_raw=data.pixels['train'][:cfg.get('train_limit')]
    state=fit_basis(train_raw,cfg['dimensions'])
    state.update(train_indices=data.indices['train'][:cfg.get('train_limit')],
                 validation_indices=data.indices['validation'][:cfg.get('validation_limit')],
                 test_indices=data.indices['test'],source_checkpoint_sha256=digest(cfg['source_checkpoint']),
                 fit_pixels_sha256=hashlib.sha256(train_raw.contiguous().numpy().tobytes()).hexdigest(),smoke=cfg['smoke'])
    full=CachedPCASpace(state,max(cfg['dimensions']))
    latents={}
    for split in ('train','validation','test'):
        raw=data.pixels[split]
        if split in ('train','validation'): raw=raw[:cfg.get(split+'_limit')]
        latents[split]=torch.cat([full.encode(x[:,None].float().div(127.5).sub(1)) for x in raw.split(1024)])
    quality=[]
    val=data.images('validation',cfg.get('validation_limit'))
    for d in cfg['dimensions']:
        space=CachedPCASpace(state,d)
        errors=[]; outside=[]
        for batch in val.split(256):
            reconstruction=space.decode(space.encode(batch))
            errors.append((reconstruction-batch).square().mean((1,2,3)))
            outside.append((reconstruction.abs()>1).float().mean((1,2,3)))
        quality.append(dict(dimension=d,representation='pca',retained_variance=float(state['explained_variance_ratio'][:d].sum()),
                            validation_reconstruction_rms=float(torch.cat(errors).mean().sqrt()),
                            reconstruction_rms_ci95=conditional_ci(torch.cat(errors).numpy(),root=True),
                            validation_per_image_reconstruction_mse=torch.cat(errors),
                            reconstructed_fraction_outside=float(torch.cat(outside).mean()),
                            train_coordinate_rms=float(latents['train'][:,:d].square().mean().sqrt()),
                            n_train=len(train_raw),n_validation=len(val),rank_estimate=state['rank_estimate'],
                            information_loss_warning=float(state['explained_variance_ratio'][:d].sum())<.9))
    save_pt(out/'latents.pt',latents);save_pt(out/'basis.pt',state)
    write_json(out/'quality.json',quality);write_json(out/'resolved_config.json',cfg)
    complete(out,sig,['basis.pt','latents.pt','quality.json','resolved_config.json'])
    return out/'basis.pt'
