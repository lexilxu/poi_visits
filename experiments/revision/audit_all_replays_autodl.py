"""Preserve diagnostic candidates and quantify the remaining replay differences."""
from pathlib import Path
from hashlib import sha256
import json
import os
import subprocess
import sys
import numpy as np

root=Path('/root/autodl-tmp/poi_continuation_20261114')
out=root/'replay_deterministic_v4_audit';out.mkdir(exist_ok=True)
base=(root/'analysis_source_v2/experiments/revision/evaluate_january_strict_v3.py').read_text()
needle='    if not strict_match:\n'
assert base.count(needle)==1
capture="""    if diagnostic_out is not None:
        np.savez_compressed(diagnostic_out.with_name(diagnostic_out.stem+'_unverified.npz'),
                            origins=origins,predictions=pred)
"""
(out/'evaluate_january_diagnostic.py').write_text(base.replace(needle,capture+needle))
driver=(root/'diagnose_replay_v4_autodl.py').read_text()
driver=driver.replace('from evaluate_january_strict_v3 import','from evaluate_january_diagnostic import')
driver=driver.replace("out=root/'replay_deterministic_v4'","out=root/'replay_deterministic_v4_audit'")
driver=driver.replace("sys.path.insert(0,str(root/'analysis_source_v2/experiments/revision'))",
    "sys.path.insert(0,str(root/'analysis_source_v2/experiments/revision'))\nsys.path.insert(0,str(root))")
(out/'driver.py').write_text(driver)
failures=json.loads((root/'new_january_results/execution_status.json').read_text())['failures']
rows=[]
for case in failures:
    recipe,seed=case['id'],case['seed'];name=f'{recipe}_seed{seed}'
    if not (out/(name+'_status.json')).exists():
        with (out/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u',str(out/'driver.py'),recipe,str(seed)],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,
                           env={**os.environ,'MKL_THREADING_LAYER':'GNU'})
    status=json.loads((out/(name+'_status.json')).read_text())
    diagnostics=json.loads((out/(name+'_differences.json')).read_text())
    selection=next(root.glob(f'v*/queue/{recipe}/selected_runs.json'))
    s=json.loads(selection.read_text());folder=selection.parent/f"seed{seed}_lr{s['selected_lr']:g}"
    with np.load(out/(name+'_differences_unverified.npz')) as new, np.load(folder/'predictions.npz') as old:
        p=new['predictions'][new['origins']<92];q=old['predicted_visits']
        y=old['observed_visits'];valid=old['valid'];test=old['dataset_split']=='test'
        absolute=np.abs(p.astype(float)-q.astype(float))
        delta_mae=float((np.abs(p[test]-y[test])-np.abs(q[test]-y[test]))[valid[test]].mean())
        rows.append(dict(recipe=recipe,status=status['status'],**diagnostics,
            absolute_difference_p999=float(np.quantile(absolute,.999)),
            december_delta_MAE=delta_mae,
            changed_activations={str(t):int(((p[test]>=t)!=(q[test]>=t))[valid[test]].sum()) for t in (.1,.25,.5,1.,2.)},
            checkpoint_sha256=status['checkpoint_sha256'],
            original_predictions_sha256=sha256((folder/'predictions.npz').read_bytes()).hexdigest()))
    (out/'audit.json').write_text(json.dumps(dict(status='in_progress',rows=rows),indent=2))
    print(json.dumps(dict(recipe=recipe,seed=seed,completed=len(rows),delta_MAE=delta_mae)),flush=True)
assert len(rows)==19
(out/'audit.json').write_text(json.dumps(dict(status='diagnostics_complete',rows=rows,
    acceptance='Candidates remain separate from accepted January results'),indent=2))
