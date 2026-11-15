from pathlib import Path
import json,tarfile
root=Path(__file__).resolve().parents[2]
out=root/'artifacts/revision_20261114/private_autodl_continuation_20261114'
with tarfile.open(out/'continuation.tar') as src, tarfile.open(out/'bootstrap.tar','x') as dst:
    for m in src:
        p=Path(m.name)
        if 'queue' in p.parts:
            if p.name not in ('manifest.json','fit.json','selected_runs.json'):continue
        if p.name.endswith('.log'):continue
        dst.addfile(m,src.extractfile(m))
print((out/'bootstrap.tar').stat().st_size)
