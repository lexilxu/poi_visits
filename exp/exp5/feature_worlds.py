"""Full EvolveGCN-H worlds for one target feature-group intervention each."""
import numpy as np
import torch
from exp.exp5.own_gradients import PHYSICAL_BATCH_SIZE

GROUPS={
    'History':None,'Location':list(range(2)),
    'Identity':list(range(4,36))+list(range(44,52)),
    'Parent':list(range(36,44)),'Others':list(range(2,4))}


def encoded_forecast(model,history,static,mean,std,physical_batch_size=PHYSICAL_BATCH_SIZE):
    count=len(history)
    if not 1<=count<=physical_batch_size:raise ValueError('Requested worlds must fit the archived physical batch')
    if static.ndim==2:static=static[None].expand(history.shape[0],-1,-1)
    assert static.shape[:2]==(history.shape[0],history.shape[2])
    history=torch.cat([history,history[-1:].expand(physical_batch_size-count,-1,-1)],dim=0)
    static=torch.cat([static,static[-1:].expand(physical_batch_size-count,-1,-1)],dim=0)
    sequence=torch.cat([history[...,None],static[:,None].expand(-1,history.shape[1],-1,-1)],dim=-1)
    for layer in model.layers:sequence=layer(sequence,model.graph)
    return torch.clamp(model.decoder(sequence[:,-1]).transpose(1,2)[:count]*std+mean,min=0.)


def perturb_group(model,history,static,targets,history_delta,static_delta,group,step,mean,std,batch_size=4,physical_batch_size=PHYSICAL_BATCH_SIZE):
    """Return H x targets; each world changes one target and recomputes weights."""
    targets=np.asarray(targets,dtype=np.int64);out=[];columns=GROUPS[group]
    if batch_size<1:raise ValueError('Use a positive intervention chunk size')
    batch_size=min(batch_size,physical_batch_size)
    with torch.no_grad():
        for start in range(0,len(targets),batch_size):
            selected=targets[start:start+batch_size];n=len(selected)
            own=history.expand(n,-1,-1).clone()
            features=static[None].expand(n,-1,-1).clone()
            for row,target in enumerate(selected):
                if columns is None:own[row,:,target]-=step*history_delta[start+row]
                else:features[row,target,columns]-=step*static_delta[start+row,columns]
            forecast=encoded_forecast(model,own,features,mean,std,physical_batch_size)
            out.append(forecast[torch.arange(n,device=forecast.device),:,torch.as_tensor(selected,device=forecast.device)].cpu().numpy())
    return np.concatenate(out).T
