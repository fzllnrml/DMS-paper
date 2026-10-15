"""Train-only PCA, unwhitened latent geometry and separate epsilon MLPs."""
import argparse
from pathlib import Path
import torch
from torch import nn
from sklearn.decomposition import PCA
from src.models.noise_schedule import NoiseSchedule
from .common import device_for,seed_all,save_pt,digest
from .data import ReferenceData

class PCASpace:
    name='pca'
    def __init__(self,state,dim):
        if not 1<=dim<=len(state['components']): raise ValueError('PCA dimension exceeds fitted rank')
        self.state=state; self.dim=dim; self.components=state['components'][:dim].float(); self.mean=state['mean'].float()
    def encode(self,images):
        x=images.flatten(1); return (x-self.mean.to(x.device))@self.components.to(x.device).T
    def decode(self,z): return (z@self.components.to(z.device)+self.mean.to(z.device)).reshape(-1,1,28,28)

def fit_pca(source_checkpoint,output,dimensions,data_root='data/mnist',download=False):
    path=Path(output); latent_path=path.with_name(path.stem+'_latents.pt')
    if path.exists() or latent_path.exists(): raise FileExistsError('PCA artifact already exists')
    data=ReferenceData(torch.load(source_checkpoint,map_location='cpu',weights_only=True),data_root,download,True)
    x=data.images('train').flatten(1)
    if not dimensions or any(type(d)!=int or not 1<=d<=min(x.shape) for d in dimensions): raise ValueError('Invalid PCA dimensions')
    estimator=PCA(n_components=max(dimensions),svd_solver='full',whiten=False).fit(x.numpy())
    state=dict(format='mnist_pca_v1',components=torch.from_numpy(estimator.components_.copy()),mean=torch.from_numpy(estimator.mean_.copy()),explained_variance=torch.from_numpy(estimator.explained_variance_.copy()),explained_variance_ratio=torch.from_numpy(estimator.explained_variance_ratio_.copy()),dimensions=sorted(set(dimensions)),train_indices=data.indices['train'],validation_indices=data.indices['validation'],source_checkpoint_sha256=digest(source_checkpoint),fit_split='train',whiten=False)
    space=PCASpace(state,max(dimensions))
    # State is written last; a partial latent file never masquerades as a complete fit.
    save_pt(latent_path,{s:space.encode(data.images(s)) for s in ('train','validation','test')}); save_pt(path,state)
    return path

class EpsilonMLP(nn.Module):
    def __init__(self,dim,hidden,T):
        super().__init__(); self.T=T; layers=[]; width=dim+1
        for n in hidden: layers.extend([nn.Linear(width,n),nn.SiLU()]); width=n
        layers.append(nn.Linear(width,dim)); self.net=nn.Sequential(*layers)
    def forward(self,x,t): return self.net(torch.cat((x,t.float()[:,None]/(self.T-1)),1))

class PCADiffusion:
    def __init__(self,dim,T=1000,beta_start=.0001,beta_end=.02,hidden=(256,256,256),device='auto'):
        if dim<1 or T<2 or not 0<beta_start<=beta_end<1: raise ValueError('Invalid dimensions/schedule')
        self.device=device_for(str(device)); self.dim=dim; self.config=dict(dim=dim,T=T,beta_start=beta_start,beta_end=beta_end,hidden=list(hidden))
        self.net=EpsilonMLP(dim,hidden,T).to(self.device); self.schedule=NoiseSchedule.linear(T,beta_start,beta_end,torch.device('cpu'))
        for k,v in vars(self.schedule).items():
            if isinstance(v,torch.Tensor): setattr(self.schedule,k,v.to(self.device))
    @torch.no_grad()
    def score(self,x,t):
        t=torch.as_tensor(t,device=self.device)
        if t.dtype not in (torch.int32,torch.int64): raise ValueError('Integer t required')
        if t.ndim==0: t=t.expand(len(x))
        if t.shape!=(len(x),) or torch.any(t<0) or torch.any(t>=self.schedule.T): raise ValueError('Invalid timestep')
        self.net.eval(); return -self.net(x,t)/self.schedule.sqrt_one_minus_alphas_cumprod[t,None]
    @torch.no_grad()
    def sample_ddim(self,n=64,num_steps=50,seed=42,clip_denoised=False,initial_noise=None):
        if clip_denoised: raise ValueError('PCA coordinates have no [-1,1] box')
        if not 1<=num_steps<=self.schedule.T or n<1: raise ValueError('Invalid DDIM parameters')
        noise=torch.randn(n,self.dim,generator=torch.Generator().manual_seed(seed)) if initial_noise is None else initial_noise
        if noise.shape!=(n,self.dim) or noise.dtype!=torch.float32 or not torch.isfinite(noise).all(): raise ValueError('Invalid noise')
        x=noise.to(self.device).clone(); self.net.eval(); indices=torch.linspace(self.schedule.T-1,0,num_steps).round().long().tolist()
        for i,t in enumerate(indices):
            a=self.schedule.alphas_cumprod[t]; prev=self.schedule.alphas_cumprod[indices[i+1]] if i+1<len(indices) else x.new_tensor(1.)
            eps=self.net(x,torch.full((n,),t,device=self.device,dtype=torch.long)); x0=(x-(1-a).sqrt()*eps)/a.sqrt(); x=prev.sqrt()*x0+(1-prev).sqrt()*eps
        return x
    @classmethod
    def from_checkpoint(cls,path,device='auto'):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state.get('format')!='mnist_pca_diffusion_v1': raise ValueError('Invalid PCA checkpoint')
        model=cls(**state['model_config'],device=device); model.net.load_state_dict(state['model']); model.net.eval(); return model

def train_pca(pca_path,output,dim,device='auto',seed=42,epochs=30,batch_size=128,lr=.0002,resume=False,model_options=None):
    from .training import fit
    path=Path(pca_path); state=torch.load(path,map_location='cpu',weights_only=True); PCASpace(state,dim)
    latents=torch.load(path.with_name(path.stem+'_latents.pt'),map_location='cpu',weights_only=True)
    seed_all(seed); model=PCADiffusion(dim=dim,device=device,**(model_options or {}))
    settings=dict(seed=seed,epochs=epochs,batch_size=batch_size,lr=lr,dim=dim,model=model.config,pca_sha=digest(path))
    template=dict(format='mnist_pca_diffusion_v1',model_config=model.config,pca=state,config={'seed':seed},split_indices={'train':state['train_indices'],'validation':state['validation_indices']})
    return fit(model.net,latents['train'][:,:dim],latents['validation'][:,:dim],output,settings,template,schedule=model.schedule,resume=resume)

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit'); f.add_argument('--checkpoint',default='runs/mnist/checkpoints/best.pt'); f.add_argument('--output',default='runs/mnist_pca/pca.pt'); f.add_argument('--dimensions',nargs='+',type=int,default=[16,32,64,128,256,512,784])
    t=sub.add_parser('train'); t.add_argument('--pca',default='runs/mnist_pca/pca.pt'); t.add_argument('--output',required=True); t.add_argument('--dim',type=int,required=True); t.add_argument('--device',default='auto'); t.add_argument('--seed',type=int,default=42); t.add_argument('--epochs',type=int,default=30); t.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.command=='fit': print(fit_pca(a.checkpoint,a.output,a.dimensions))
    else: print(train_pca(a.pca,a.output,a.dim,a.device,a.seed,a.epochs,resume=a.resume))
if __name__=='__main__': main()
