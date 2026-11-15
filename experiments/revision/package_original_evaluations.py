from pathlib import Path
import tarfile,hashlib,json
root=Path(__file__).resolve().parents[2];base=root/'artifacts/revision_20261114'
out=base/'private_autodl_continuation_20261114/original_evaluations.tar'
files=[]
with tarfile.open(out,'x') as tar:
    for name in ('run_records','january','attribution','attribution_january'):
        for p in (base/'neural_results'/name).rglob('*'):
            if p.is_file():
                arc=str(p.relative_to(base/'neural_results')).replace('\\','/')
                tar.add(p,arcname=arc,recursive=False)
                files.append(dict(path=arc,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
out.with_suffix('.json').write_text(json.dumps(dict(files=files,archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
print(dict(files=len(files),bytes=out.stat().st_size))
