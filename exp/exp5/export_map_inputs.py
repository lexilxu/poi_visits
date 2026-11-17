"""Export mapped counts and positions on AutoDL4 for the original spatial figure."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,socket
import numpy as np
import pandas as pd


def main(args):
    record=json.loads((args.tables/'complete.json').read_text())
    assert record['status']=='complete'
    profiles=Path(record['profile_source']);original=profiles/'poi_profiles.csv'
    assert sha256(original.read_bytes()).hexdigest()==record['profile_sha256']
    metadata=json.loads((profiles/'complete.json').read_text())
    assert metadata['status']=='complete' and metadata['checkpoint_sha256']==record['checkpoint_sha256']
    spatial=args.tables/'attribution_cluster_assignments.csv';frame=pd.read_csv(spatial)
    own=pd.read_csv(original)
    np.testing.assert_array_equal(frame.placekey,own.placekey)
    np.testing.assert_array_equal(frame.regime,own.regime)
    names=['History-driven','Location-driven','Identity-driven','Parent-driven','Others-driven']
    mapping={name:i for i,name in enumerate(names)};mapping['Unassigned']=-1
    assert set(frame.regime).issubset(mapping)
    points=frame[['placekey','longitude','latitude']].copy()
    points['regime']=frame.regime.map(mapping).astype(int)
    bounds=[-122.520,-122.185,47.490,47.764]
    assert np.isfinite(points[['longitude','latitude']].to_numpy()).all()
    points['shown']=points.longitude.between(*bounds[:2])&points.latitude.between(*bounds[2:])
    counts={name:int((points.shown&(points.regime==i)).sum()) for i,name in enumerate(names)}
    missing=int((points.shown&(points.regime==-1)).sum());mapped=int(points.shown.sum())
    assert sum(counts.values())+missing==mapped
    args.out.mkdir(parents=True,exist_ok=False);path=args.out/'map_points.csv';points.to_csv(path,index=False)
    report=dict(status='complete',period='December 2020',protocol=metadata['protocol'],seed=metadata['seed'],
        forecast_origins=metadata['forecast_origins'],bounds=bounds,regime_counts=counts,mapped_unassigned=missing,
        mapped_POIs=mapped,outside_extent=len(points)-mapped,checkpoint_sha256=record['checkpoint_sha256'],
        spatial_source_sha256=sha256(spatial.read_bytes()).hexdigest(),profile_sha256=record['profile_sha256'],
        points_sha256=sha256(path.read_bytes()).hexdigest(),code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--tables',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
