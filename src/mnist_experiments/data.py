"""Reuse diffusion train/validation indices. Test references are explicit opt-in."""
import torch
from torchvision.datasets import MNIST
class ReferenceData:
    def __init__(self,checkpoint,root='data/mnist',download=False,include_test=False):
        train=MNIST(root,train=True,download=download); split=checkpoint.get('split_indices')
        if split is None: raise ValueError('Checkpoint must record train/validation split_indices')
        self.indices={k:torch.as_tensor(split[k]).long().cpu() for k in ('train','validation')}
        if set(self.indices['train'].tolist()) & set(self.indices['validation'].tolist()): raise ValueError('Overlapping splits')
        for ids in self.indices.values():
            if not len(ids) or ids.min()<0 or ids.max()>=len(train.data) or len(torch.unique(ids))!=len(ids): raise ValueError('Invalid split indices')
        self.pixels={k:train.data[ids] for k,ids in self.indices.items()}; self.targets={k:train.targets[ids] for k,ids in self.indices.items()}
        if include_test:
            test=MNIST(root,train=False,download=download); self.pixels['test']=test.data; self.targets['test']=test.targets; self.indices['test']=torch.arange(len(test.data))
    def images(self,split,limit=None): return self.pixels[split][:limit].unsqueeze(1).float().div(127.5).sub(1)
    def labels(self,split,limit=None): return self.targets[split][:limit]
