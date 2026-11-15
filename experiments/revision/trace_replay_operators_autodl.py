"""Isolate repeat-forward variability using one archived input batch."""
from pathlib import Path
import json
import sys
import torch

root=Path('/root/autodl-tmp/poi_continuation_20261114')
sys.path.insert(0,str(root/'analysis_source_v2/experiments/revision'))
import train_controlled
import adaptive_mtgnn
from evaluate_january_strict_v3 import extended_panel,replay_model
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
recipe,seed=sys.argv[1],int(sys.argv[2])
selection=next(root.glob(f'v*/queue/{recipe}/selected_runs.json'))
selected=json.loads(selection.read_text())
folder=selection.parent/f"seed{seed}_lr{selected['selected_lr']:g}"
out=root/'replay_operator_trace_2049';out.mkdir(exist_ok=True)
destination=out/f'{recipe}_seed{seed}.json'
assert not destination.exists()
rows=[]
seen=set()

def compare(name,a,b):
    if a.layout==torch.sparse_coo:
        a=a.coalesce();b=b.coalesce()
        same=torch.equal(a.indices(),b.indices())
        rows.append(dict(operator=name,identical_indices=same))
        if not same:return
        a,b=a.values(),b.values()
    delta=(a-b).abs()
    rows.append(dict(operator=name,shape=list(a.shape),
                     max_difference=delta.max().item(),mean_difference=delta.mean().item(),
                     differing_values=(delta>0).sum().item()))

def traced(name,operation):
    def call(*args,**kwargs):
        first=operation(*args,**kwargs)
        if name not in seen:
            seen.add(name)
            second=operation(*args,**kwargs)
            compare(name,first,second)
        return first
    return call

original_mm=torch.sparse.mm
def sparse_mm(a,b,*args,**kwargs):
    return traced('sparse_mm_'+str(a.layout)+'_'+str(tuple(b.shape)),original_mm)(a,b,*args,**kwargs)
torch.sparse.mm=sparse_mm
adaptive_mtgnn.normalize_with_self=traced('adaptive_normalization',adaptive_mtgnn.normalize_with_self)
original_make=train_controlled.make_model
class ProbeFinished(Exception):pass
def make(*args,**kwargs):
    model=original_make(*args,**kwargs)
    for name,module in model.named_modules():
        if hasattr(module,'multiply'):
            module.multiply=traced(name+'.graph_multiply',module.multiply)
        if isinstance(module,adaptive_mtgnn.SparseGraphConstructor):
            module.forward=traced(name+'.graph_constructor',module.forward)
        if isinstance(module,(torch.nn.Conv2d,torch.nn.LayerNorm)):
            module.forward=traced(name+'.'+type(module).__name__,module.forward)
    def done(module,inputs,output):
        destination.write_text(json.dumps(dict(recipe=recipe,seed=seed,operators=rows,
            scope='Two calls to each operator on identical inputs; first archived evaluation batch'),indent=2))
        raise ProbeFinished()
    model.register_forward_hook(done)
    return model
train_controlled.make_model=make
raw,dates,attrs=extended_panel(Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip'),
                              root/'private_validation_202101/extended_panel.npz')
torch.manual_seed(seed)
try:
    replay_model(folder,raw,dates,attrs,'cuda:0',root/'private_graphs')
except ProbeFinished:
    print(json.dumps(dict(recipe=recipe,seed=seed,
        differing_operators=[r for r in rows if r.get('differing_values',0)>0 or r.get('identical_indices') is False])),flush=True)
