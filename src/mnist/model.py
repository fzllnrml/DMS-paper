"""Stage-one checkpoint-compatible epsilon U-Net, plus explicit DDIM x_T."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.noise_schedule import NoiseSchedule

class TimeEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer('frequencies',torch.exp(-math.log(10000)*torch.arange(dim//2)/(dim//2-1)))
        self.mlp=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
    def forward(self,t):
        a=t.float()[:,None]*self.frequencies[None,:]
        return self.mlp(torch.cat((a.sin(),a.cos()),1))

class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,time_dim):
        super().__init__()
        self.norm1=nn.GroupNorm(8,cin); self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.time=nn.Linear(time_dim,cout)
        self.norm2=nn.GroupNorm(8,cout); self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)
    def forward(self,x,emb):
        h=self.conv1(F.silu(self.norm1(x)))+self.time(F.silu(emb))[:,:,None,None]
        return self.conv2(F.silu(self.norm2(h)))+self.skip(x)

class EpsilonUNet(nn.Module):
    def __init__(self,base_channels=32,time_dim=128):
        super().__init__()
        if base_channels<8 or base_channels%8 or time_dim<4 or time_dim%2: raise ValueError('Invalid widths')
        c=base_channels
        self.time=TimeEmbedding(time_dim); self.input=nn.Conv2d(1,c,3,padding=1)
        self.enc1=ResidualBlock(c,c,time_dim); self.down1=nn.Conv2d(c,2*c,4,stride=2,padding=1)
        self.enc2=ResidualBlock(2*c,2*c,time_dim); self.down2=nn.Conv2d(2*c,4*c,4,stride=2,padding=1)
        self.mid1=ResidualBlock(4*c,4*c,time_dim); self.mid2=ResidualBlock(4*c,4*c,time_dim)
        self.up2=nn.Conv2d(4*c,2*c,3,padding=1); self.dec2=ResidualBlock(4*c,2*c,time_dim)
        self.up1=nn.Conv2d(2*c,c,3,padding=1); self.dec1=ResidualBlock(2*c,c,time_dim)
        self.output=nn.Sequential(nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,1,3,padding=1))
    def forward(self,x,t):
        emb=self.time(t); h1=self.enc1(self.input(x),emb); h2=self.enc2(self.down1(h1),emb)
        h=self.mid2(self.mid1(self.down2(h2),emb),emb)
        h=self.up2(F.interpolate(h,size=h2.shape[-2:],mode='nearest'))
        h=self.dec2(torch.cat((h,h2),1),emb)
        h=self.up1(F.interpolate(h,size=h1.shape[-2:],mode='nearest'))
        return self.output(self.dec1(torch.cat((h,h1),1),emb))

class MNISTDiffusion:
    dim=784
    image_shape=(1,28,28)
    def __init__(self,T=1000,beta_start=1e-4,beta_end=.02,base_channels=32,time_dim=128,device='auto'):
        if type(T)!=int or T<2 or not 0<beta_start<=beta_end<1: raise ValueError('Invalid schedule')
        if str(device)=='auto': device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device=torch.device(device)
        if self.device.type not in {'cpu','mps','cuda'}: raise ValueError('Unsupported device')
        if self.device.type=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
        if self.device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        self.config=dict(T=T,beta_start=beta_start,beta_end=beta_end,base_channels=base_channels,time_dim=time_dim)
        self.schedule=NoiseSchedule.linear(T,beta_start,beta_end,torch.device('cpu'))
        for k,v in vars(self.schedule).items():
            if isinstance(v,torch.Tensor): setattr(self.schedule,k,v.to(self.device))
        self.net=EpsilonUNet(base_channels,time_dim).to(self.device); self._count_nfe=False; self._nfe=0
    def _images(self,x):
        if x.dtype!=torch.float32 or x.device!=next(self.net.parameters()).device: raise ValueError('x must be float32 on model.device')
        if x.ndim==1 and x.shape==(784,): return x.reshape(1,1,28,28)
        if x.ndim==2 and x.shape[1]==784: return x.reshape(-1,1,28,28)
        if x.ndim==4 and tuple(x.shape[1:])==self.image_shape: return x
        raise ValueError('Expected [B,1,28,28], [B,784], or [784]')
    def _times(self,t,n):
        t=torch.as_tensor(t,device=self.device)
        if t.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8): raise ValueError('Integer diffusion indices required')
        if t.ndim==0: t=t.expand(n)
        if t.shape!=(n,) or torch.any(t<0) or torch.any(t>=self.schedule.T): raise ValueError('Invalid timestep batch')
        return t.long()
    @contextmanager
    def _evaluation(self):
        training=self.net.training; self.net.eval()
        try:
            with torch.no_grad(): yield
        finally: self.net.train(training)
    def _epsilon(self,x,t):
        if self._count_nfe: self._nfe+=len(x)
        return self.net(x,t)
    def q_sample(self,x0,t,noise):
        x=self._images(x0); t=self._times(t,len(x)); s=self.schedule
        if noise.shape!=x0.shape: raise ValueError('Noise shape mismatch')
        return (s.extract(s.sqrt_alphas_cumprod,t,x.shape)*x+s.extract(s.sqrt_one_minus_alphas_cumprod,t,x.shape)*self._images(noise)).reshape(x0.shape)
    def score(self,x_batch,t_batch):
        x=self._images(x_batch); t=self._times(t_batch,len(x))
        with self._evaluation(): return (-self._epsilon(x,t)/self.schedule.extract(self.schedule.sqrt_one_minus_alphas_cumprod,t,x.shape)).reshape(x_batch.shape)
    def score_numpy(self,x,t): return self.score(torch.as_tensor(x,dtype=torch.float32,device=self.device),t).cpu().numpy()
    def enable_nfe_counting(self): self._count_nfe=True; self.reset_nfe()
    def reset_nfe(self): self._nfe=0
    @property
    def nfe(self): return self._nfe
    def sample_ddim(self,n=64,num_steps=50,seed=42,clip_denoised=True,initial_noise=None):
        """eta=0; optional explicit float32 [n,1,28,28] noise preserves paired starts."""
        if type(n)!=int or n<1 or type(num_steps)!=int or not 1<=num_steps<=self.schedule.T: raise ValueError('Invalid n/steps')
        if initial_noise is None: x=torch.randn((n,*self.image_shape),generator=torch.Generator().manual_seed(seed)).to(self.device)
        else:
            if initial_noise.shape!=(n,*self.image_shape) or initial_noise.dtype!=torch.float32 or not torch.isfinite(initial_noise).all(): raise ValueError('Invalid initial_noise')
            x=initial_noise.to(self.device).clone()
        indices=torch.linspace(self.schedule.T-1,0,num_steps).round().long().tolist()
        with self._evaluation():
            for i,current in enumerate(indices):
                t=torch.full((n,),current,device=self.device,dtype=torch.long)
                a=self.schedule.alphas_cumprod[current]
                prev=self.schedule.alphas_cumprod[indices[i+1]] if i+1<len(indices) else x.new_tensor(1.)
                eps=self._epsilon(x,t); x0=(x-(1-a).sqrt()*eps)/a.sqrt()
                if clip_denoised:
                    x0=x0.clamp(-1,1); eps=(x-a.sqrt()*x0)/(1-a).sqrt()
                x=prev.sqrt()*x0+(1-prev).sqrt()*eps
        return x
    @classmethod
    def from_checkpoint(cls,path,device='auto'):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state.get('format')!='mnist_diffusion_v1': raise ValueError('Invalid MNIST checkpoint')
        model=cls(**state['model_config'],device=device); model.net.load_state_dict(state['model']); model.net.eval()
        return model
