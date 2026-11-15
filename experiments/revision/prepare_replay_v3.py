"""Match inference allocation and normalization to the saved training evaluator."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
base=root/'artifacts/revision_20261114/private_autodl_continuation_20261114'
source=(base/'evaluate_january_strict_v2.py').read_text(encoding='utf8')
source=source.replace("torch.cuda.set_per_process_memory_fraction(.4 if long_input else .25)","torch.cuda.set_per_process_memory_fraction(1.0)")
needle="    pred=np.empty((len(origins),h,len(attrs)),dtype='float32')"
source=source.replace(needle,"    arrays=[torch.as_tensor((v-saved['visit_mean'])/saved['visit_std'],device=device,dtype=torch.float32) for v in (graph,own)]\n"+needle)
old="""inputs=[torch.as_tensor((v[ids]-saved['visit_mean'])/saved['visit_std'],
                                         device=device,dtype=torch.float32) for v in (graph,own)]"""
assert old in source
source=source.replace(old,"inputs=[v[ids] for v in arrays]")
old="""inputs=[torch.as_tensor((v[ids]-saved['visit_mean'])/saved['visit_std'],
                               device=device,dtype=torch.float32) for v in (graph,own)]"""
assert old in source
source=source.replace(old,"inputs=[v[ids] for v in arrays]")
source=source.replace("if long_input and device.startswith('cuda'):torch.cuda.empty_cache()","# Retain the fitted evaluator allocation pattern between batches.")
(base/'evaluate_january_strict_v3.py').write_text(source,encoding='utf8')
