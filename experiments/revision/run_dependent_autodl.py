from pathlib import Path
import json,sys,shutil,hashlib,tarfile,subprocess,time
from types import SimpleNamespace
r=Path('/root/autodl-tmp/poi_continuation_20261114');out=r/'dependent_20261114'
out.mkdir(exist_ok=False)
manifest=json.loads((r/'dependent_inputs.json').read_text())
assert hashlib.sha256((r/'dependent_inputs.tar').read_bytes()).hexdigest()==manifest['archive_sha256']
with tarfile.open(r/'dependent_inputs.tar') as tar:tar.extractall(out,filter='data')
for e in manifest['files']:assert hashlib.sha256((out/e['path']).read_bytes()).hexdigest()==e['sha256']
source=out/'source';shutil.copytree(r/'v2/source',source)
scripts=source/'experiments/revision'
for p in (out/'scripts').glob('*.py'):shutil.copy2(p,scripts/p.name)
sys.path.insert(0,str(scripts))
from paired_seed_analysis import collected,pairs
bundle=Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip')
queues=[r/'v1/queue',r/'v2/queue']
args=SimpleNamespace(queues=queues,baselines=out/'baselines',january=None)
groups=collected(args)
cache=out/'uncertainty_december/comparisons';cache.mkdir(parents=True)
reused=0
for model,reference in pairs(groups):
    name=model.replace(' ','_')+'__vs__'+reference.replace(' ','_')+'.json'
    previous=out/'prior_comparisons'/name
    if not previous.exists():continue
    p=json.loads(previous.read_text())
    hashes=lambda key:[hashlib.sha256(x.read_bytes()).hexdigest() for x in groups[key]['paths']]
    assert hashes(model)==p['model_predictions'] and hashes(reference)==p['reference_predictions'],name
    shutil.copy2(previous,cache/name);reused+=1
print(json.dumps(dict(reused_verified_comparisons=reused)),flush=True)
base=['--queues',*[str(q) for q in queues],'--bundle',str(bundle),'--baselines',str(out/'baselines')]
for name,script in [('density','summarize_density.py'),('coverage_december','spatial_priority_analysis.py'),('uncertainty_december','paired_seed_analysis.py')]:
    with (out/(name+'.log')).open('x') as log:
        result=subprocess.run([sys.executable,'-u',str(scripts/script),*base,'--out',str(out/name)],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
    if result.returncode:raise RuntimeError(f'{name} failed; inspect log')
    (out/(name+'_complete.json')).write_text(json.dumps(dict(status='complete',finished_at=time.time())))
    print(json.dumps(dict(completed=name)),flush=True)
