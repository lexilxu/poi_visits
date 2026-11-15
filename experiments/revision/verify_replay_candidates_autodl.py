"""Independently repeat all diagnostic forecasts before a protocol decision."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import numpy as np
root=Path('/root/autodl-tmp/poi_continuation_20261114')
first=root/'replay_deterministic_v4_audit'
audit=json.loads((first/'audit.json').read_text())
assert audit['status']=='diagnostics_complete'
out=root/'replay_deterministic_v4_repeat';out.mkdir(exist_ok=False)
shutil.copy2(first/'evaluate_january_diagnostic.py',out/'evaluate_january_diagnostic.py')
(out/'driver.py').write_text((first/'driver.py').read_text().replace(
    "out=root/'replay_deterministic_v4_audit'","out=root/'replay_deterministic_v4_repeat'"))
rows=[]
for row in audit['rows']:
    name=f"{row['recipe']}_seed{row['seed']}"
    with (out/(name+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-u',str(out/'driver.py'),row['recipe'],str(row['seed'])],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,
            env={**os.environ,'MKL_THREADING_LAYER':'GNU'})
    filename=name+'_differences_unverified.npz'
    with np.load(first/filename) as a,np.load(out/filename) as b:
        np.testing.assert_array_equal(a['origins'],b['origins'])
        delta=np.abs(a['predictions'].astype(float)-b['predictions'].astype(float))
        before=a['origins']<92
        rows.append(dict(recipe=row['recipe'],seed=row['seed'],
                         december_max_difference=float(delta[before].max()),
                         january_max_difference=float(delta[~before].max()),
                         bit_identical=bool(np.array_equal(a['predictions'],b['predictions']))))
    (out/'repeat_audit.json').write_text(json.dumps(dict(status='in_progress',rows=rows),indent=2))
    print(json.dumps(dict(completed=len(rows),**rows[-1])),flush=True)
(out/'repeat_audit.json').write_text(json.dumps(dict(status='complete',rows=rows,
    all_predictions_bit_identical=all(r['bit_identical'] for r in rows)),indent=2))
