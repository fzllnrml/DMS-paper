"""Shared small epoch trainer for the independent evaluator and PCA diffusion."""
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from .common import save_pt,write_json,seed_all,fingerprint

def fit(net,train_x,val_x,output,settings,template,train_y=None,val_y=None,schedule=None,resume=False):
    output=Path(output); latest=output/'latest.pt'; device=next(net.parameters()).device
    if latest.exists() and not resume: raise FileExistsError('Training exists: use --resume')
    if settings['epochs']<1 or settings['batch_size']<1: raise ValueError('Invalid epochs/batch_size')
    optimizer=torch.optim.AdamW(net.parameters(),lr=settings['lr'],weight_decay=0)
    order=torch.Generator().manual_seed(settings['seed']+1); rng=torch.Generator().manual_seed(settings['seed']+2)
    signature=fingerprint({k:v for k,v in settings.items() if k!='epochs'})
    start,best,history=0,float('inf'),[]
    if resume and latest.exists():
        old=torch.load(latest,map_location='cpu',weights_only=True)
        if old['training_signature']!=signature: raise ValueError('Training resume settings changed')
        net.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer']); order.set_state(old['order_rng']); rng.set_state(old['noise_rng'])
        start,best,history=old['epoch'],old['best_loss'],old['history']
    loader=DataLoader(TensorDataset(train_x,train_y if train_y is not None else torch.zeros(len(train_x))),batch_size=settings['batch_size'],shuffle=True,generator=order)
    def loss_for(x,y,generator):
        x=x.to(device)
        if schedule is None:
            logits=net(x)[0]; y=y.to(device)
            return F.cross_entropy(logits,y),int((logits.argmax(1)==y).sum())
        t=torch.randint(schedule.T,(len(x),),generator=generator).to(device)
        noise=torch.randn(x.shape,generator=generator).to(device); a=schedule.alphas_cumprod[t,None]
        return F.mse_loss(net(a.sqrt()*x+(1-a).sqrt()*noise,t),noise),0
    for epoch in range(start+1,settings['epochs']+1):
        net.train(); total,count=0.,0
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True); loss,_=loss_for(x,y,rng)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            total+=loss.item()*len(x); count+=len(x)
        val_rng=torch.Generator().manual_seed(settings['seed']+3); net.eval(); loss_sum,correct,nval=0.,0,0
        with torch.no_grad():
            for begin in range(0,len(val_x),settings['batch_size']):
                x=val_x[begin:begin+settings['batch_size']]; y=val_y[begin:begin+len(x)] if val_y is not None else None
                loss,ncorrect=loss_for(x,y,val_rng); loss_sum+=loss.item()*len(x); correct+=ncorrect; nval+=len(x)
        validation=loss_sum/nval; improved=validation<best; best=min(best,validation)
        row=dict(epoch=epoch,train_loss=total/count,validation_loss=validation)
        if schedule is None: row['validation_accuracy']=correct/nval
        history.append(row)
        state=dict(**template,model=net.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best_loss=best,history=history,
                   order_rng=order.get_state(),noise_rng=rng.get_state(),training_signature=signature,settings=settings)
        if schedule is None: state['metadata']=dict(template['metadata'],validation_accuracy=correct/nval)
        save_pt(latest,state)
        if improved: save_pt(output/'best.pt',state)
        write_json(output/'history.json',history)
    return output/'best.pt'
