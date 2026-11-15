"""Acquire official payment-based occupancy records at prespecified clock times."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import date,timedelta
from hashlib import sha256
from pathlib import Path
import gzip
import json
import time
import requests


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    tasks=[]
    start=date(2020,7,1)
    while start<date(2021,2,1):
        end=min(start+timedelta(days=7),date(start.year+1,1,1),date(2021,2,1))
        tasks.append((start,end));start=end
    def one(span):
        start,end=span;destination=a.out/f'{start}_{end}.json.gz'
        receipt=destination.with_suffix('.receipt.json')
        if receipt.exists():
            old=json.loads(receipt.read_text(encoding='utf8'))
            assert sha256(destination.read_bytes()).hexdigest()==old['sha256']
            return
        assert not destination.exists(),destination
        timestamps=[];day=start
        while day<end:
            timestamps += [f"'{day}T{hour}:00:00'" for hour in ('10','14','18')]
            day+=timedelta(days=1)
        dataset='wtpb-jp8d' if start.year==2020 else 'jb6y-98nr'
        url=f'https://cos-data.seattle.gov/resource/{dataset}.json'
        params={'$select':'occupancydatetime,paidoccupancy,sourceelementkey,parkingspacecount,paidparkingarea,paidparkingsubarea,parkingcategory,paidparkingrate,location',
                '$where':'occupancydatetime in ('+','.join(timestamps)+')',
                '$order':'occupancydatetime,sourceelementkey','$limit':'50000'}
        session=requests.Session();session.trust_env=False
        response=session.get(url,params=params,timeout=(15,240));response.raise_for_status()
        records=response.json();assert isinstance(records,list) and len(records)<50000
        payload=json.dumps(records,separators=(',',':')).encode()
        compressed=gzip.compress(payload,mtime=0)
        destination.write_bytes(compressed)
        report=dict(dataset=dataset,url=url,query=params,start_inclusive=str(start),end_exclusive=str(end),
                    records=len(records),bytes=len(compressed),sha256=sha256(compressed).hexdigest(),
                    response_sha256=sha256(response.content).hexdigest(),retrieved_at=time.time(),
                    definition='Payment-based occupancy; observations at 10:00, 14:00 and 18:00 local clock time')
        receipt.write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps(report),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,tasks))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
