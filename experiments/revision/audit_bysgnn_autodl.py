"""Measure the official BysGNN attention bottleneck on AutoDL4."""
from pathlib import Path
from hashlib import sha1, sha256
import gc
import json
import time
import torch

root = Path('/root/autodl-tmp/poi_continuation_20261114')
source = root/'bysgnn_resource_2044/base_model_exact.py'
payload = source.read_bytes()
blob = sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
assert blob == '1499e74a0f475a0e5530f555e7211db678a7ab22', blob
out = source.parent/'resource_probe.json'
assert not out.exists()
torch.set_num_threads(2)
torch.manual_seed(42)
device = 'cuda:0'
properties = torch.cuda.get_device_properties(0)
report = dict(source_blob_sha=blob, source_sha256=sha256(payload).hexdigest(),
              gpu=properties.name, gpu_bytes=properties.total_memory,
              torch=torch.__version__, batch_size=4, nodes=9884, gru_dim=128,
              heads=8, dropout=.5, purpose='Synthetic resource probe; no fitted result',
              started_at=time.time())
# This is the exact official multihead-attention module and hidden-input shape.
# The per-POI recurrent weights are allocated to represent the official network.
# Other layers, recurrent activations, gradients and optimizer states add memory.
try:
    attention=torch.nn.MultiheadAttention(128,8,.5,device=device,batch_first=True).train()
    small=torch.randn(4,32,128,device=device,requires_grad=True)
    values,weights=attention(small,small,small)
    (values.square().mean()+weights.square().mean()).backward()
    report['small_attention_forward_backward']='passed'
    del small,values,weights
    attention.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    recurrent=torch.nn.ModuleList(torch.nn.GRU(128,128,batch_first=True,device=device) for _ in range(9884))
    report['recurrent_parameters']=sum(p.numel() for p in recurrent.parameters())
    assert report['recurrent_parameters']==979227648
    torch.cuda.reset_peak_memory_stats()
    hidden=torch.randn(4,9884,128,device=device,requires_grad=True)
    report['phase']='official_attention_forward'
    values,weights=attention(hidden,hidden,hidden)
    report['forward_shape']=[list(values.shape),list(weights.shape)]
    report['phase']='official_attention_backward'
    (values.square().mean()+weights.square().mean()).backward()
    report['status']='attention_probe_passed_full_model_feasibility_pending'
except torch.OutOfMemoryError as error:
    report.update(status='out_of_memory',error=str(error))
finally:
    report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),finished_at=time.time())
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
