from pathlib import Path
import json,hashlib,tarfile,sys,shutil,subprocess
r=Path('/root/autodl-tmp/poi_continuation_20261114')
original=r/'original_evaluations';original.mkdir(exist_ok=False)
m=json.loads((r/'original_evaluations.json').read_text())
assert hashlib.sha256((r/'original_evaluations.tar').read_bytes()).hexdigest()==m['archive_sha256']
with tarfile.open(r/'original_evaluations.tar') as tar:tar.extractall(original,filter='data')
for e in m['files']:assert hashlib.sha256((original/e['path']).read_bytes()).hexdigest()==e['sha256']
scripts=r/'dependent_20261114/source/experiments/revision'
sys.path.insert(0,str(scripts))
from summarize_queue import merge
out=r/'consolidated_20261114';out.mkdir(exist_ok=False)
for period in ('december','january'):
    destination=out/period;cache=destination/'run_records';cache.mkdir(parents=True)
    prior=original/('run_records' if period=='december' else 'january/run_records')
    current=r/('new_december_results_v2/run_records' if period=='december' else 'new_january_results/run_records')
    for source in (prior,current):
        for p in source.glob('*.json'):
            dest=cache/p.name
            assert not dest.exists(),dest
            shutil.copy2(p,dest)
    if period=='january':
        pred=destination/'private_predictions';pred.mkdir()
        for source in (original/'january/private_predictions',r/'new_january_results/private_predictions'):
            for p in source.glob('*.npz'):
                if not (cache/(p.stem+'.json')).exists():continue
                assert not (pred/p.name).exists()
                (pred/p.name).symlink_to(p)
    merge(cache,destination)
    print(json.dumps(dict(consolidated=period,records=len(list(cache.glob('*.json'))))),flush=True)
bundle='/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip'
for period,folder in [('december','attribution'),('january','attribution_january')]:
    command=[sys.executable,'-u',str(scripts/'summarize_attribution.py'),'--queue',str(r/'v1/queue'),'--bundle',bundle,
             '--attribution',str(original/folder),'--out',str(out/('parent_'+period))]
    if period=='january':command+=['--january',str(out/'january')]
    with (out/('parent_'+period+'.log')).open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    print(json.dumps(dict(completed='parent_'+period)),flush=True)
