"""Package executable sources for the authorized AutoDL run; exclude all data."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED
import json

ROOT=Path(__file__).resolve().parents[2]


def package(destination):
    files=[]
    for directory in ('experiments/revision','experiments/archived_2020q4','src','external/MTGNN'):
        files.extend(p for p in (ROOT/directory).rglob('*.py') if '__pycache__' not in p.parts)
    if not files:
        raise RuntimeError('No training source files found')
    members={p.relative_to(ROOT).as_posix():p for p in files}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,'w',compression=ZIP_DEFLATED) as z:
        for name,path in sorted(members.items()):
            z.write(path,name)
    manifest={'archive':destination.name,'sha256':sha256(destination.read_bytes()).hexdigest(),
              'members':{name:sha256(p.read_bytes()).hexdigest() for name,p in sorted(members.items())},
              'contains_data':False}
    destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'archive':str(destination),'files':len(members),'sha256':manifest['sha256']}))
    return manifest


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/revision_20261114/training_source.zip')
    args=p.parse_args(); package(args.out)
