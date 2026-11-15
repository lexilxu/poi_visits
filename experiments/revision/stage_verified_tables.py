from pathlib import Path
import hashlib,json,tarfile,shutil
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'artifacts/revision_20261114/private_autodl_continuation_20261114'
archive=base/'evidence_2031.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='a595e866640b5bb7a5f5ce26b821fb9cbed2f94a954f333b4a414a16034fa31c'
out=ROOT/'artifacts/e2031';out.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
results=out/'results';results.mkdir()
for period in ('december','january'):
    target=results if period=='december' else results/'january';target.mkdir(exist_ok=True)
    for p in (out/'consolidated_20261114'/period).glob('*'):
        if p.is_file():shutil.copy2(p,target/p.name)
shutil.copytree(out/'dependent_20261114/density',results/'density')
backup=out/'paper_before';shutil.copytree(ROOT/'paper/revision',backup)
print(results)
