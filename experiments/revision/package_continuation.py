"""Archive the existing frozen runs and inputs for an AutoDL continuation."""
from pathlib import Path
import hashlib, json, tarfile

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/revision_20261114'
OUT = ART / 'private_autodl_continuation_20261114'
OUT.mkdir(exist_ok=True)
archive = OUT / 'continuation.tar'
files = []
for version in ('v1', 'v2'):
    base = ART / 'private_local_runs' / ('20261114_' + version)
    for path in base.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            files.append((path, str(Path(version) / path.relative_to(base)).replace('\\', '/')))
for folder in ('private_graphs', 'private_extension_inputs', 'private_validation_202101'):
    for path in (ART / folder).rglob('*'):
        if path.is_file():
            files.append((path, str(Path(folder) / path.relative_to(ART / folder)).replace('\\', '/')))
records = []
with tarfile.open(archive, 'x') as tar:
    for path, name in files:
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        records.append(dict(path=name, size=path.stat().st_size, sha256=digest))
        tar.add(path, arcname=name, recursive=False)
manifest = dict(files=records, archive_sha256=hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest())
(OUT / 'transfer_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(files=len(records), bytes=archive.stat().st_size, sha256=manifest['archive_sha256'])), flush=True)
