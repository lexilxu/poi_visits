"""Construct cohorts from July records, before August validation/September test."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import numpy as np


def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    sources={}
    for month in ('2020-07','2020-08','2020-09'):
        p=a.inputs/(month+'_city_visits.npz')
        with np.load(p) as z:sources[month]={k:z[k] for k in z.files}
    report=[]
    for city,region in [('Seattle','WA'),('Portland','OR')]:
        july=sources['2020-07'];ids=np.sort(july['poi_id'][(july['city']==city)&(july['region']==region)])
        arrays=[];coverage=[]
        for month,data in sources.items():
            ix={v:i for i,v in enumerate(data['poi_id'])};values=np.full((data['visits'].shape[0],len(ids)),np.nan,np.float32)
            present=[(i,ix[k]) for i,k in enumerate(ids) if k in ix];dest,src=zip(*present)
            values[:,list(dest)]=data['visits'][:,list(src)];arrays.append(values)
            coverage.append(dict(month=month,present=len(present),missing=len(ids)-len(present)))
        raw=np.concatenate(arrays);assert raw.shape==(92,len(ids)) and np.isfinite(raw[:31]).all()
        path=a.out/(city+'.npz')
        np.savez_compressed(path,visits=raw,dates=np.arange('2020-07-01','2020-10-01',dtype='datetime64[D]').astype(str),poi_id=ids,city=city)
        report.append(dict(city=city,POIs=len(ids),coverage=coverage,panel_sha256=sha256(path.read_bytes()).hexdigest(),
                           cohort_rule='All city records in July; subsequent missing rows remain missing'))
    (a.out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
