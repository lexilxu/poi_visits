"""Continue frozen recipes, preserving completed fits and source snapshots."""
from pathlib import Path
import hashlib, importlib.util, json, os, subprocess, sys, time
from types import SimpleNamespace

ROOT = Path('/root/autodl-tmp/poi_continuation_20261114')
PANEL = Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip')

def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def prepare(bootstrap=False):
    import tarfile
    manifest = json.loads((ROOT/'transfer_manifest.json').read_text())
    if not bootstrap and (ROOT/'reconciliation.json').exists():
        assert digest(ROOT/'continuation.tar') == manifest['archive_sha256']
        expected={e['path']:e['sha256'] for e in manifest['files']}
        completed={str(Path(p).parent) for p in expected if p.endswith('/fit.json')}
        with tarfile.open(ROOT/'continuation.tar') as tar:
            for member in tar:
                parts=Path(member.name).parts
                name=member.name
                if len(parts)>=4 and parts[1]=='queue' and parts[3].startswith('seed') and '/'.join(parts[:4]) not in completed:
                    name=str(Path('interrupted_evidence').joinpath(*parts[2:]))
                dest=ROOT/name
                assert dest.resolve().is_relative_to(ROOT)
                if dest.exists():
                    assert digest(dest)==expected[member.name], name
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as out,tar.extractfile(member) as inp:
                        import shutil
                        shutil.copyfileobj(inp,out)
                    assert digest(dest)==expected[member.name], name
        save(ROOT/'hydration_complete.json',dict(files=len(expected),archive_sha256=manifest['archive_sha256'],completed_at=time.time()))
        print('Full checkpoint archive verified and hydrated. The existing training process continues.',flush=True)
        # The initial transfer launcher must finish before its historical launch step.
        sys.exit(10)
    archive=ROOT/('bootstrap.tar' if bootstrap else 'continuation.tar')
    if not bootstrap:
        assert digest(archive) == manifest['archive_sha256']
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (ROOT/member.name).resolve()
            assert target.is_relative_to(ROOT) and not target.exists(), target
        tar.extractall(ROOT, filter='data')
    for entry in manifest['files']:
        if (ROOT/entry['path']).exists():
            assert digest(ROOT/entry['path']) == entry['sha256'], entry['path']
        else:
            assert bootstrap
    assert digest(PANEL) == '12d6414990eeb5703b1179152b5993518dcfa68361e39e09ae621919950695ec'
    local = {}
    for path in (ROOT/'v1/queue').glob('*/seed*/fit.json'):
        fit = json.loads(path.read_text())
        assert fit['status'] == 'complete'
        for name in ('checkpoint.pt','predictions.npz','epoch_log.json'):
            name_in_manifest=str((path.parent/name).relative_to(ROOT))
            assert any(e['path']==name_in_manifest for e in manifest['files'])
        local[(path.parent.parent.name, fit['seed'], fit['lr'])] = fit
    remote = []
    old = Path('/root/autodl-tmp/poi_revision_20261114a/private_runs/core')
    for path in old.glob('*/seed*/fit.json'):
        fit = json.loads(path.read_text())
        key = (path.parent.parent.name, fit['seed'], fit['lr'])
        remote.append(dict(recipe=key[0],seed=key[1],lr=key[2],overlap=key in local,
                           bundle_sha256=fit['bundle_sha256']))
    assert len(local) == 115
    assert len(remote) == 84 and all(r['overlap'] for r in remote)
    partial=[]
    for path in (ROOT/'v1/queue').glob('*/seed*'):
        if path.is_dir() and not (path/'fit.json').exists():
            target=ROOT/'interrupted_evidence'/path.parent.name/path.name
            target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists()
            path.rename(target)
            partial.append(str(target.relative_to(ROOT)))
    save(ROOT/'reconciliation.json', dict(canonical_complete_fits=len(local),
         canonical_complete_recipes=len(list((ROOT/'v1/queue').glob('*/selected_runs.json'))),
         remote_prior_fits=remote,interrupted_evidence=partial,remaining_fits=382,
         policy='Retain the existing local result cohort; retain all remote prior runs as separate evidence.'))
    print('Verified transfer, 115 canonical fits, 84 overlapping remote fits; preserved interrupted evidence.',flush=True)

def run():
    import fcntl
    lock=(ROOT/'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start=time.monotonic()
    for version,stage in (('v1','all'),('v2','extension')):
        base=ROOT/version; source=base/'source'; queue=base/'queue'
        directory=source/'experiments/revision'
        sys.path.insert(0,str(directory))
        spec=importlib.util.spec_from_file_location('frozen_queue_'+version,directory/'training_queue.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        plan=module.plan(stage)
        if version=='v1':
            existing=json.loads((queue/'manifest.json').read_text())
            assert plan['recipes']==existing['recipes']
            for name,expected in existing['source_sha256'].items():
                assert digest(source/name)==expected, name
        save(base/'continuation_plan.json',plan)
        args=SimpleNamespace(bundle=PANEL,graph_dir=ROOT/'private_graphs',
             extra_input_dir=ROOT/'private_extension_inputs',epochs=100,batch_size=4,device='cuda:0')
        subprocess.run([sys.executable,str(directory/'train_controlled.py'),'--smoke','--device','cuda:0'],check=True,cwd=source)
        def one(recipe,seed,lr):
            dest=queue/recipe['id']/f'seed{seed}_lr{lr:g}'
            record=dest/'fit.json'
            if record.exists():
                fit=json.loads(record.read_text());assert fit['status']=='complete'
                return {**fit,'directory':str(dest)}
            assert not dest.exists(), f'Inspect partial fit: {dest}'
            assert time.monotonic()-start < 48*3600, '48-hour continuation budget reached'
            dest.parent.mkdir(parents=True,exist_ok=True)
            command=module.command(args,recipe,seed,lr,dest)
            logfile=dest.parent/(dest.name+'.autodl_continuation.log')
            print(json.dumps(dict(starting=recipe['id'],seed=seed,lr=lr,time=time.time())),flush=True)
            with logfile.open('x') as stream:
                subprocess.run(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
            fit=json.loads(record.read_text());assert fit['status']=='complete'
            return {**fit,'directory':str(dest)}
        for recipe in plan['recipes']:
            tuning=[one(recipe,42,lr) for lr in module.LEARNING_RATES]
            chosen=module.selected_fit(tuning)
            selected=[chosen]+[one(recipe,seed,chosen['lr']) for seed in module.SEEDS[1:]]
            selection=queue/recipe['id']/'selected_runs.json'
            if selection.exists():
                previous=json.loads(selection.read_text())
                assert previous['selected_lr']==chosen['lr']
                assert sorted(r['seed'] for r in previous['runs'])==list(module.SEEDS)
            else:
                save(selection,dict(recipe=recipe,selected_lr=chosen['lr'],selection_metric='seed42 validation MAE',
                     runs=[{k:f[k] for k in ('seed','lr','validation_MAE','directory')} for f in selected]))
        save(base/'continuation_complete.json',dict(completed_at=time.time(),recipes=len(plan['recipes'])))
        sys.path.remove(str(directory))
    save(ROOT/'training_complete.json',dict(completed_at=time.time(),recipes=71,fits=497))

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='bootstrap': prepare(True)
    elif sys.argv[1]=='run': run()
