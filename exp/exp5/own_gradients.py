"""Exact own-POI Jacobian blocks for identity-graph EvolveGCN-H.

With identity convolution, a source node affects other predictions only when
selected by one of the hard top-k summaries. Sum-output gradients are therefore
diagonal blocks outside that union. Inside the union we compute individual
output VJPs. This reduces work without estimating or dropping cross-node terms.
"""
import torch

PHYSICAL_BATCH_SIZE=4


def forward_with_support(model,history,static,physical_batch_size=PHYSICAL_BATCH_SIZE):
    if history.shape[0]!=1:raise ValueError('Use one forecast origin for exact source-node support')
    if type(model.graph).__name__!='IdentityGraph':raise ValueError('Support shortcut requires identity graph convolution')
    supports=[];handles=[]
    def capture(module,args,output):
        nodes=args[0]
        scores=(nodes@module.scorer)/module.scorer.norm()
        ids=scores.squeeze(-1).topk(min(module.k,nodes.shape[1]),dim=1).indices[0]
        supports.append(ids)
    for layer in model.layers:
        handles.append(layer.evolve_weights.choose_topk.register_forward_hook(capture))
    try:
        sequence=torch.cat([history.expand(physical_batch_size,-1,-1)[...,None],
            static[None,None].expand(physical_batch_size,history.shape[1],-1,-1)],dim=-1)
        for layer in model.layers:sequence=layer(sequence,model.graph)
        prediction=model.decoder(sequence[:,-1]).transpose(1,2)[:1]
    finally:
        for handle in handles:handle.remove()
    return prediction,torch.unique(torch.cat(supports))


def diagonal_blocks(prediction,history,static,support,lead,chunk_size=8):
    """Return d Y[j] / d history[:,j] and d Y[j] / d static[j,:]."""
    output=prediction[:,lead,:]
    gh,gs=torch.autograd.grad(output.sum(),(history,static),retain_graph=True)
    dh,ds=gh.clone(),gs.clone()
    for ids in support.split(chunk_size):
        selectors=output.new_zeros((len(ids),*output.shape))
        selectors[torch.arange(len(ids),device=ids.device),0,ids]=1
        bh,bs=torch.autograd.grad(output,(history,static),grad_outputs=selectors,
                                  is_grads_batched=True,retain_graph=True)
        for k,j in enumerate(ids):
            dh[0,:,j]=bh[k,0,:,j]
            ds[j]=bs[k,j]
    return dh,ds
