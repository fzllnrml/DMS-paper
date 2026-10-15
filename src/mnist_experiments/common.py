"""Serialization, reproducibility, code hashes and image-counted inference."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,os,random,subprocess
from pathlib import Path
import numpy as np
import torch

def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)

def device_for(name):
    if name=='auto': name='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(name)

def plain(v):
    if isinstance(v,torch.Tensor): return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,Path): return str(v)
    if isinstance(v,dict): return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [plain(x) for x in v]
    return v

def safe_tensors(v):
    if isinstance(v,torch.Tensor): return v.detach().cpu()
    if isinstance(v,np.ndarray): return torch.from_numpy(v.copy())
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {k:safe_tensors(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [safe_tensors(x) for x in v]
    return v

def write_json(path,v):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(plain(v),indent=2,allow_nan=False)); tmp.replace(p)

def save_pt(path,v):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    torch.save(safe_tensors(v),tmp); tmp.replace(p)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()

def fingerprint(v): return hashlib.sha256(json.dumps(plain(v),sort_keys=True).encode()).hexdigest()[:16]
def timestamp(): return datetime.now(timezone.utc).isoformat()
def merge_dict(a,b):
    result=deepcopy(a)
    for k,v in b.items():
        if k=='schedule' and isinstance(v,dict) and v.get('kind')!=result.get(k,{}).get('kind'): result[k]=deepcopy(v)
        else: result[k]=merge_dict(result[k],v) if isinstance(v,dict) and isinstance(result.get(k),dict) else deepcopy(v)
    return result

def code_digest():
    folder=Path(__file__).parent; h=hashlib.sha256()
    for p in sorted(folder.glob('*.py'))+[folder.parent/'mnist/model.py']:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()

def git_revision():
    try: return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)))
    except (OSError,subprocess.CalledProcessError): return dict(commit=None,dirty=None)

def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)
    if device.type=='mps': torch.mps.synchronize()

class PixelSpace:
    dim=784; name='pixel'
    def encode(self,x): return x.flatten(1)
    def decode(self,x): return x.reshape(-1,1,28,28)

class CountedModel:
    def __init__(self,model,batch_size=128):
        if batch_size<2: raise ValueError('batch_size must be >=2')
        self.model=model; self.batch_size=batch_size; self.device=model.device; self.dim=model.dim; self.schedule=model.schedule
        self.nfe=0; self.forward_calls=0
    def score(self,x,t):
        t=torch.as_tensor(t,device=self.device)
        if t.ndim==0: t=t.expand(len(x))
        if t.shape!=(len(x),): raise ValueError('Timestep shape mismatch')
        out=[]
        for start in range(0,len(x),self.batch_size):
            batch=x[start:start+self.batch_size]; out.append(self.model.score(batch,t[start:start+len(batch)]))
            self.nfe+=len(batch); self.forward_calls+=1
        out=torch.cat(out)
        if not torch.isfinite(out).all(): raise FloatingPointError('Nonfinite score')
        return out
    def ddim(self,noise,steps):
        from src.mnist import MNISTDiffusion
        result=[]
        for batch in noise.split(self.batch_size):
            if isinstance(self.model,MNISTDiffusion): batch=batch.reshape(-1,1,28,28)
            result.append(self.model.sample_ddim(n=len(batch),num_steps=steps,initial_noise=batch).flatten(1))
            self.nfe+=len(batch)*steps; self.forward_calls+=steps
        return torch.cat(result)

def gaussian_noise(n,dim,seed):
    # CPU random draws, not per-particle neural calls. Prefix stable for every n.
    rng=torch.Generator().manual_seed(seed)
    return torch.stack([torch.randn(dim,generator=rng) for _ in range(n)])
