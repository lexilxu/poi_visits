"""Verify a raw gzip source and package the fixed cohort and two cities."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import gzip
import io
import json
import sys
import time
import zipfile
import pandas as pd


class HashedReader:
    def __init__(self, stream):self.stream=stream;self.digest=sha256();self.bytes=0
    def read(self, size=-1):
        value=self.stream.read(size);self.digest.update(value);self.bytes+=len(value);return value


def run(a):
    assert not a.out.exists(),a.out
    a.out.mkdir(parents=True)
    with zipfile.ZipFile(a.bundle) as z:
        names=[n for n in z.namelist() if n.endswith('poi_attributes.csv')];assert len(names)==1
        attrs=pd.read_csv(z.open(names[0]),dtype=str)
    ids=set(attrs.placekey)
    fields={'placekey','safegraph_place_id','parent_placekey','parent_safegraph_place_id','location_name',
            'street_address','city','region','postal_code','brands','safegraph_brand_ids','date_range_start',
            'date_range_end','raw_visit_counts','raw_visitor_counts','visits_by_day','device_type'}
    source=a.input.open('rb') if a.input else sys.stdin.buffer
    reader=HashedReader(source);raw=gzip.GzipFile(fileobj=reader,mode='rb')
    count=0;retained=0;first=True;months=set();cities={};cohort_matches=0
    destination=a.out/'selected_monthly_rows.csv.gz'
    with gzip.open(destination,'wt',encoding='utf8',newline='') as output:
        for chunk in pd.read_csv(raw,usecols=lambda c:c in fields,dtype=str,chunksize=100000):
            assert {'placekey','city','region','date_range_start','date_range_end','visits_by_day'}<=set(chunk)
            count+=len(chunk);months.update(chunk.date_range_start.str[:7].unique())
            cohort=chunk.placekey.isin(ids)
            city=chunk.city.fillna('').str.casefold();region=chunk.region.fillna('').str.upper()
            keep=cohort|((city=='seattle')&(region=='WA'))|((city=='portland')&(region=='OR'))
            selected=chunk.loc[keep];retained+=len(selected);cohort_matches+=int(cohort.sum())
            if len(selected):
                selected.to_csv(output,index=False,header=first);first=False
                for key,n in selected.groupby(['city','region']).size().items():cities['/'.join(key)]=cities.get('/'.join(key),0)+int(n)
    raw.close()
    assert reader.read(1)==b''
    assert reader.bytes==a.bytes and reader.digest.hexdigest()==a.sha256
    if a.input:source.close()
    assert months=={a.month},months
    report=dict(status='complete',month=a.month,source_id=a.source_id,source_sha256=a.sha256,source_bytes=reader.bytes,
                gzip_crc_verified=True,source_rows=count,retained_rows=retained,fixed_cohort_rows=cohort_matches,cities=cities,
                cohort_rule='Original 9884 placekeys plus Seattle WA and Portland OR city records; outcomes play no role in selection.',
                output_sha256=sha256(destination.read_bytes()).hexdigest(),completed_at=time.time())
    (a.out/'complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--input',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--bytes',type=int,required=True)
    p.add_argument('--month',required=True);p.add_argument('--source-id',required=True);run(p.parse_args())
