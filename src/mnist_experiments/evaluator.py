"""Independent frozen classifier; never trained implicitly by a search job."""
import argparse
import torch
from torch import nn
from .common import device_for,seed_all,digest
from .data import ReferenceData

class EvaluatorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,64),nn.ReLU())
        self.classifier=nn.Linear(64,10)
    def forward(self,x):
        h=self.body(x); return self.classifier(h),h

class FrozenEvaluator:
    def __init__(self,model,device='cpu',batch_size=256,metadata=None):
        self.device=device_for(str(device)); self.model=model.to(self.device).eval(); self.model.requires_grad_(False)
        self.batch_size=batch_size; self.metadata=metadata or {}; self.images_evaluated=0; self.forward_calls=0
    @classmethod
    def load(cls,path,device='auto',batch_size=256):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state.get('format')!='mnist_evaluator_v1': raise ValueError('Invalid evaluator checkpoint')
        model=EvaluatorCNN(); model.load_state_dict(state['model'])
        return cls(model,device,batch_size,state['metadata'])
    @torch.no_grad()
    def evaluate(self,images):
        logits=[]; features=[]
        for batch in images.split(self.batch_size):
            y,h=self.model(batch.to(self.device).reshape(-1,1,28,28)); logits.append(y.cpu()); features.append(h.cpu())
            self.images_evaluated+=len(batch); self.forward_calls+=1
        return torch.cat(logits),torch.cat(features)
    def features(self,images): return self.evaluate(images)[1]

def train_evaluator(source_checkpoint,output,device='auto',seed=71,epochs=5,batch_size=128,lr=.001,data_root='data/mnist',download=False,resume=False,limit=None):
    from .training import fit
    data=ReferenceData(torch.load(source_checkpoint,map_location='cpu',weights_only=True),data_root,download)
    seed_all(seed); net=EvaluatorCNN().to(device_for(device))
    settings=dict(seed=seed,epochs=epochs,batch_size=batch_size,lr=lr,limit=limit,source_sha=digest(source_checkpoint))
    template=dict(format='mnist_evaluator_v1',metadata=dict(seed=seed,train_indices=data.indices['train'],validation_indices=data.indices['validation'],limit=limit,labels_used=True))
    return fit(net,data.images('train',limit),data.images('validation',limit),output,settings,template,data.labels('train',limit),data.labels('validation',limit),resume=resume)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--diffusion-checkpoint',default='runs/mnist/checkpoints/best.pt'); p.add_argument('--output',default='runs/mnist_evaluator')
    p.add_argument('--device',default='auto'); p.add_argument('--seed',type=int,default=71); p.add_argument('--epochs',type=int,default=5); p.add_argument('--resume',action='store_true')
    a=p.parse_args(); print(train_evaluator(a.diffusion_checkpoint,a.output,a.device,a.seed,a.epochs,resume=a.resume))
if __name__=='__main__': main()
