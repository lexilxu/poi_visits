"""Download the identified shared monthly files and preserve exact source bytes."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
import json
import time
import requests


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action=None; self.params={}; self.inside=False
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if tag=='form' and attrs.get('id')=='download-form':
            self.action=attrs.get('action'); self.inside=True
        if self.inside and tag=='input' and attrs.get('type')=='hidden':
            self.params[attrs['name']]=attrs.get('value','')
    def handle_endtag(self, tag):
        if tag=='form':self.inside=False


def download(row, root):
    folder=root/row['month'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/row['name'];receipt=path.with_suffix(path.suffix+'.json')
    if receipt.exists():
        info=json.loads(receipt.read_text());assert path.stat().st_size==info['bytes']==row['bytes']
        h=sha256()
        with path.open('rb') as f:
            for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
        assert h.hexdigest()==info['sha256'];return info
    assert not path.exists() and not path.with_suffix(path.suffix+'.part').exists(),path
    session=requests.Session();session.trust_env=False
    response=session.get('https://drive.google.com/uc',params={'export':'download','id':row['id']},stream=True,timeout=(30,120))
    response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type',''):
        form=DownloadForm();form.feed(response.text);response.close()
        assert form.action and urlparse(form.action).hostname=='drive.usercontent.google.com'
        assert form.params.get('id')==row['id']
        response=session.get(form.action,params=form.params,stream=True,timeout=(30,120));response.raise_for_status()
    assert 'text/html' not in response.headers.get('Content-Type',''), 'Expected shared gzip download'
    partial=path.with_suffix(path.suffix+'.part');h=sha256();count=0;last=time.monotonic()
    with partial.open('xb') as f:
        for data in response.iter_content(4*1024*1024):
            if count==0:assert data[:2]==b'\x1f\x8b'
            f.write(data);h.update(data);count+=len(data)
            assert count<=row['bytes']
            if time.monotonic()-last>25:
                print(json.dumps(dict(month=row['month'],file=row['name'],downloaded=count,total=row['bytes'])),flush=True);last=time.monotonic()
    response.close();assert count==row['bytes'];partial.rename(path)
    info=dict(**row,sha256=h.hexdigest(),status='downloaded_size_and_sha256_verified',downloaded_at=time.time())
    receipt.write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True);return info


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=json.loads(a.manifest.read_text())['files']
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda row:download(row,a.out),rows))
