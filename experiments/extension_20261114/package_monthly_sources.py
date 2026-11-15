"""Input packaging only: deterministic CSV selection, gzip CRC and source hashes."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from extract_monthly_stream import run


def main(a):
    rows=json.loads(a.manifest.read_text(encoding='utf8'))['files']
    def one(row):
        path=a.raw/row['month']/row['name']
        info=json.loads(path.with_suffix(path.suffix+'.json').read_text(encoding='utf8'))
        out=a.out/row['month']/row['name'].removesuffix('.csv.gz')
        if (out/'complete.json').exists():
            old=json.loads((out/'complete.json').read_text(encoding='utf8'))
            assert old['source_sha256']==info['sha256']
            return
        run(Namespace(bundle=a.bundle,out=out,input=path,sha256=info['sha256'],
                      bytes=row['bytes'],month=row['month'],source_id=row['id']))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,rows))


if __name__=='__main__':
    p=ArgumentParser()
    for field in ('manifest','raw','out','bundle'):p.add_argument('--'+field,type=Path,required=True)
    main(p.parse_args())
