"""Create an immutable private backup of the completed AutoDL extension."""
from pathlib import Path
from hashlib import sha256
import json
import tarfile
import time

roots=[Path('/root/autodl-tmp/poi_extension_20261114'),Path('/root/poi_extension_analysis_20261114')]
out=Path('/root/poi_extension_evidence_20261114.tar.gz')
manifest=out.with_name('poi_extension_evidence_20261114.json')
assert not out.exists() and not manifest.exists()
files=[]
for root in roots:
    for path in sorted(root.rglob('*')):
        if path.is_file():
            name=root.name+'/'+path.relative_to(root).as_posix()
            files.append((path,dict(path=name,bytes=path.stat().st_size,sha256=sha256(path.read_bytes()).hexdigest())))
with tarfile.open(out,'x:gz',compresslevel=3) as tar:
    for path,record in files:tar.add(path,arcname=record['path'],recursive=False)
report=dict(status='complete',archive_bytes=out.stat().st_size,archive_sha256=sha256(out.read_bytes()).hexdigest(),
            files=[r for _,r in files],completed_at=time.time())
manifest.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
