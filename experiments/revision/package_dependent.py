from pathlib import Path
import tarfile,hashlib,json
root=Path(__file__).resolve().parents[2];a=root/'artifacts/revision_20261114'
out=a/'private_autodl_continuation_20261114/dependent_inputs.tar'
entries=[]
with tarfile.open(out,'x') as tar:
    paths=[(p,'baselines/'+p.name) for p in (a/'complete_horizon/private_predictions').glob('*_H7.npz')]
    paths += [(p,'prior_comparisons/'+p.name) for p in (a/'neural_results/uncertainty/comparisons').glob('*.json')]
    for name in ('summarize_queue.py','summarize_density.py','spatial_priority_analysis.py','paired_seed_analysis.py'):
        paths.append((root/'experiments/revision'/name,'scripts/'+name))
    for p,name in paths:
        tar.add(p,arcname=name,recursive=False)
        entries.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest=dict(files=entries,archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
out.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
print(dict(files=len(entries),bytes=out.stat().st_size))
