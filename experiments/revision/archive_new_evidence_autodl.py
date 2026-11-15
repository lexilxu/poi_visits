"""Archive new selected checkpoints, predictions, and complete fit records."""
from pathlib import Path,PureWindowsPath
from hashlib import sha256
import json
import tarfile
import time
root=Path('/root/autodl-tmp/poi_continuation_20261114')
prior={x['path']:x['sha256'] for x in json.loads((root/'transfer_manifest.json').read_text())['files']}
selected=set()
for selection in root.glob('v*/queue/*/selected_runs.json'):
    selected.add(selection)
    info=json.loads(selection.read_text())
    assert sorted(x['seed'] for x in info['runs'])==[42,43,44,45,46]
    for run in info['runs']:
        folder=selection.parent/PureWindowsPath(run['directory']).name
        for name in ('checkpoint.pt','predictions.npz','fit.json','epoch_log.json'):
            p=folder/name;assert p.is_file(),p
            selected.add(p)
for pattern in ('v*/queue/*/*/fit.json','v*/queue/*/*/epoch_log.json'):
    selected.update(root.glob(pattern))
for folder in ('new_january_results','january_numerical_revision_20261114','new_december_results_v2',
               'consolidated_final_20261114','replay_deterministic_v4_audit','replay_deterministic_v4_repeat'):
    for p in (root/folder).rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.csv','.npz','.py'):
            selected.add(p)
rows=[];reused=0
for p in sorted(selected):
    relative=p.relative_to(root).as_posix()
    digest=sha256(p.read_bytes()).hexdigest()
    if prior.get(relative)==digest:
        reused+=1;continue
    rows.append(dict(path=relative,size=p.stat().st_size,sha256=digest))
manifest=root/'new_evidence_backup_manifest.json'
assert not manifest.exists()
manifest.write_text(json.dumps(dict(files=rows,previous_archive_files_reused=reused,
    total_bytes=sum(r['size'] for r in rows),created_at=time.time()),indent=2))
archive=root/'new_evidence_backup_20261114.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    tar.add(manifest,arcname=manifest.name)
    for row in rows:tar.add(root/row['path'],arcname=row['path'])
digest=sha256()
with archive.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
complete=dict(status='complete',files=len(rows),uncompressed_bytes=sum(r['size'] for r in rows),
              archive_bytes=archive.stat().st_size,archive_sha256=digest.hexdigest(),completed_at=time.time())
(root/'new_evidence_backup_complete.json').write_text(json.dumps(complete,indent=2))
print(json.dumps(complete),flush=True)
