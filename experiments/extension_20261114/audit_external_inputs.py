"""Verify acquired sources and construct analysis panels exclusively on AutoDL."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import gzip
import json
import sys
import tarfile
import numpy as np
import pandas as pd


def main(a):
    manifest=json.loads(a.archive.with_suffix('.json').read_text())
    assert sha256(a.archive.read_bytes()).hexdigest()==manifest['sha256']
    a.out.mkdir(parents=True,exist_ok=False)
    sources=a.out/'sources';sources.mkdir()
    with tarfile.open(a.archive) as t:t.extractall(sources,filter='data')
    for name,digest in manifest['files'].items():assert sha256((sources/name).read_bytes()).hexdigest()==digest,name
    sys.path.insert(0,str(a.source/'experiments/revision'))
    from audit_protocol import load_panel
    old,old_dates,attrs=load_panel(a.bundle)
    month_rows=[];monthly=[]
    for month in ('2020-07','2020-08','2020-09'):
        shards=[]
        for folder in sorted((sources/'selected'/month).iterdir()):
            receipt=json.loads((folder/'complete.json').read_text())
            p=folder/'selected_monthly_rows.csv.gz'
            assert receipt['gzip_crc_verified'] and receipt['output_sha256']==sha256(p.read_bytes()).hexdigest()
            shards.append(pd.read_csv(p,dtype=str))
        data=pd.concat(shards,ignore_index=True);assert data.placekey.notna().all()
        assert not data.placekey.duplicated().any(),month
        day_count=31 if month!='2020-09' else 30
        values=np.array([json.loads(s) for s in data.visits_by_day],dtype=np.float32)
        assert values.shape==(len(data),day_count) and np.isfinite(values).all() and (values>=0).all()
        missing=set(attrs.placekey)-set(data.placekey)
        lookup={name:i for i,name in enumerate(data.placekey)}
        panel=np.full((day_count,len(attrs)),np.nan,dtype=np.float32)
        present=[(j,lookup[name]) for j,name in enumerate(attrs.placekey) if name in lookup]
        js,ks=zip(*present);panel[:,list(js)]=values[list(ks)].T
        monthly.append(panel)
        month_rows.append(dict(month=month,selected_rows=len(data),original_cohort_present=len(present),
                               original_cohort_missing=len(missing),city_rows=data.groupby(['city','region']).size().to_dict()))
        # JSON serializes string keys; city names are aggregate metadata.
        month_rows[-1]['city_rows']={'/'.join(k):int(v) for k,v in month_rows[-1]['city_rows'].items()}
        np.savez_compressed(a.out/(month+'_city_visits.npz'),visits=values.T,poi_id=data.placekey.to_numpy(str),
                            city=data.city.to_numpy(str),region=data.region.to_numpy(str))
    long_visits=np.concatenate(monthly+[old])
    long_dates=np.arange('2020-07-01','2021-01-01',dtype='datetime64[D]').astype(str)
    assert long_visits.shape==(184,9884)
    np.savez_compressed(a.out/'july_december_original_cohort.npz',visits=long_visits,dates=long_dates,poi_id=attrs.placekey.to_numpy(str))
    records=[]
    for p in sorted((sources/'parking_raw').glob('*.json.gz')):
        receipt=json.loads(p.with_suffix('.receipt.json').read_text())
        assert sha256(p.read_bytes()).hexdigest()==receipt['sha256']
        rows=json.loads(gzip.decompress(p.read_bytes()));assert len(rows)==receipt['records']
        records+=rows
    parking=pd.DataFrame(records)
    parking['timestamp']=pd.to_datetime(parking.occupancydatetime)
    parking['date']=parking.timestamp.dt.strftime('%Y-%m-%d')
    parking['hour']=parking.timestamp.dt.hour
    assert set(parking.hour)<={10,14,18}
    assert not parking.duplicated(['timestamp','sourceelementkey']).any()
    for field in ('paidoccupancy','parkingspacecount'):
        parking[field]=pd.to_numeric(parking[field],errors='coerce')
    parking['longitude']=parking.location.apply(lambda x:float(x['coordinates'][0]) if isinstance(x,dict) else np.nan)
    parking['latitude']=parking.location.apply(lambda x:float(x['coordinates'][1]) if isinstance(x,dict) else np.nan)
    valid=(parking.paidoccupancy>=0)&(parking.parkingspacecount>0)&np.isfinite(parking.latitude)&np.isfinite(parking.longitude)
    train=parking.date.between('2020-10-01','2020-11-19')&valid
    train_data=parking.loc[train]
    eligible=train_data.groupby(['sourceelementkey','hour']).date.nunique()
    eligible=eligible[eligible>=15]
    block_ids=sorted(set(eligible.index.get_level_values(0)))
    geometry=train_data.groupby('sourceelementkey')[['latitude','longitude','parkingspacecount']].median().loc[block_ids]
    geometry.to_csv(a.out/'parking_train_geometry.csv')
    date_strings=np.arange('2020-07-01','2021-02-01',dtype='datetime64[D]').astype(str)
    di={v:i for i,v in enumerate(date_strings)};bi={v:i for i,v in enumerate(block_ids)};hi={10:0,14:1,18:2}
    shape=(len(date_strings),3,len(block_ids));occupancy=np.full(shape,np.nan,np.float32);capacity=occupancy.copy()
    allowed=set(eligible.index)
    for row in parking.loc[valid].itertuples():
        if (row.sourceelementkey,row.hour) not in allowed:continue
        ix=(di[row.date],hi[row.hour],bi[row.sourceelementkey])
        occupancy[ix]=row.paidoccupancy;capacity[ix]=row.parkingspacecount
    np.savez_compressed(a.out/'parking_panel.npz',occupancy=occupancy,capacity=capacity,dates=date_strings,
                        hours=np.array([10,14,18]),block_id=np.array(block_ids),
                        latitude=geometry.latitude.to_numpy(),longitude=geometry.longitude.to_numpy(),
                        training_capacity=geometry.parkingspacecount.to_numpy())
    daily=parking.groupby('date').agg(records=('sourceelementkey','size'),blockfaces=('sourceelementkey','nunique'))
    daily.to_csv(a.out/'parking_daily_coverage.csv')
    report=dict(status='complete',source_archive_sha256=manifest['sha256'],monthly=month_rows,
                parking=dict(source_rows=len(parking),invalid_rows=int((~valid).sum()),
                             first_date=parking.date.min(),last_date=parking.date.max(),observed_days=int(len(daily)),
                             training_blockfaces=len(block_ids),training_blockface_hour_pairs=len(eligible),
                             valid_panel_observations=int(np.isfinite(occupancy).sum()),
                             paidoccupancy_above_capacity=int((parking.paidoccupancy>parking.parkingspacecount).sum()),
                             cohort_rule='At least 15 observed training dates per blockface and clock time during October 1-November 19; positive capacity and finite coordinates',
                             label='Payment-based occupancy, with missing source observations retained as missing'))
    (a.out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for f in ('archive','out','source','bundle'):p.add_argument('--'+f,type=Path,required=True)
    main(p.parse_args())
