"""Archive explicitly selected executable sources and a protocol before fitting."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import zipfile


def main(a):
    root=Path(__file__).resolve().parents[2]
    files=[]
    for folder in ('experiments/revision','experiments/archived_2020q4','external/MTGNN','src','experiments/extension_20261114'):
        files+=list((root/folder).glob('*.py'))
    files.append(Path(__file__).with_name('PROTOCOL.md'))
    assert not a.out.exists()
    with zipfile.ZipFile(a.out,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):z.write(path,path.relative_to(root).as_posix())
    report=dict(archive_sha256=sha256(a.out.read_bytes()).hexdigest(),files={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(dict(sha256=report['archive_sha256'],files=len(files))))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
