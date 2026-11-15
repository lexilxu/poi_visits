"""Stage one verified source at a time, extract on AutoDL, and retain local raw files."""
from argparse import ArgumentParser
from pathlib import Path
import json
import subprocess
import time


def run(a):
    prefix=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','10150','root@connect.cqa1.seetacloud.com']
    remote='/root/autodl-tmp/poi_extension_20261114'
    subprocess.run(prefix+['mkdir -p '+remote+'/source '+remote+'/monthly '+remote+'/incoming'],check=True)
    subprocess.run(['scp','-P','10150',str(Path(__file__).with_name('extract_monthly_stream.py')),
                    'root@connect.cqa1.seetacloud.com:'+remote+'/source/extract_monthly_stream.py'],check=True)
    rows=json.loads(a.manifest.read_text())['files']
    for row in rows:
        path=a.raw/row['month']/row['name'];receipt=path.with_suffix(path.suffix+'.json')
        started=time.monotonic()
        while not receipt.exists():
            if time.monotonic()-started>7200:raise TimeoutError(str(receipt))
            time.sleep(5)
        info=json.loads(receipt.read_text());assert info['bytes']==row['bytes'] and path.stat().st_size==row['bytes']
        out=remote+'/monthly/'+row['month']+'/'+row['name'].removesuffix('.csv.gz')
        previous=subprocess.run(prefix+['cat '+out+'/complete.json'],capture_output=True,text=True)
        if previous.returncode==0:
            prior=json.loads(previous.stdout);assert prior['source_sha256']==info['sha256']
            print(json.dumps(dict(reused=out,source_sha256=info['sha256'])),flush=True);continue
        command=['/root/miniconda3/bin/python',remote+'/source/extract_monthly_stream.py',
                 '--bundle','/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip','--out',out,
                 '--sha256',info['sha256'],'--bytes',str(row['bytes']),'--month',row['month'],'--source-id',row['id']]
        incoming=remote+'/incoming/'+row['month']+'_'+row['name']
        print(json.dumps(dict(staging=row['month']+'/'+row['name'],bytes=row['bytes'])),flush=True)
        subprocess.run(['scp','-P','10150',str(path),'root@connect.cqa1.seetacloud.com:'+incoming],check=True)
        result=subprocess.run(prefix+[' '.join(command)+' < '+incoming],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr[-2000:]+result.stdout[-2000:])
        report=json.loads(result.stdout.strip().splitlines()[-1]);assert report['status']=='complete'
        receipt.with_name(receipt.name+'.autodl.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        # The exact raw bytes and their receipt remain in local durable storage.
        # The extractor has just verified the complete remote input hash and CRC.
        assert report['source_sha256']==info['sha256'] and report['source_bytes']==row['bytes']
        subprocess.run(prefix+['rm -- '+incoming],check=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
    run(p.parse_args())
