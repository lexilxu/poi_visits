"""Check deterministic graph evaluation against existing strict replay bounds."""
from pathlib import Path
from hashlib import sha256
import json
import sys
import time
import torch
root=Path('/root/autodl-tmp/poi_continuation_20261114')
sys.path.insert(0,str(root/'analysis_source_v2/experiments/revision'))
import train_controlled
from evaluate_january_strict_v3 import extended_panel,replay_model
from deterministic_graph_replay import install,validate
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
device=torch.device('cuda:0')
validate(device);install()
recipe,seed=sys.argv[1],int(sys.argv[2])
selection=next(root.glob(f'v*/queue/{recipe}/selected_runs.json'))
s=json.loads(selection.read_text())
folder=selection.parent/f"seed{seed}_lr{s['selected_lr']:g}"
out=root/'replay_deterministic_v4';out.mkdir(exist_ok=True)
name=f'{recipe}_seed{seed}'
assert not (out/(name+'_status.json')).exists()
torch.manual_seed(seed)
raw,dates,attrs=extended_panel(Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip'),root/'private_validation_202101/extended_panel.npz')
status=dict(recipe=recipe,seed=seed,operator_validation='passed',
            operator_source_sha256=sha256((root/'deterministic_graph_replay.py').read_bytes()).hexdigest(),
            checkpoint_sha256=sha256((folder/'checkpoint.pt').read_bytes()).hexdigest(),
            scope='Diagnostic replay with original tolerances; accepted results remain unchanged',started_at=time.time())
try:
    origins,pred,extra=replay_model(folder,raw,dates,attrs,'cuda:0',root/'private_graphs',out/(name+'_differences.json'))
    import numpy as np
    np.savez_compressed(out/(name+'_candidate.npz'),origins=origins,predictions=pred)
    status.update(status='replay_passed',**extra)
except Exception as error:
    status.update(status='replay_failed',error=str(error))
status['finished_at']=time.time()
(out/(name+'_status.json')).write_text(json.dumps(status,indent=2))
print(json.dumps(status),flush=True)
