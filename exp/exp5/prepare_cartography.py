"""Prepare public 2020 TIGER/Line geometry, without any POI records."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import io
import json
import sys
from zipfile import ZipFile
import numpy as np


def run(args):
    if args.pyshp_wheel:sys.path.insert(0,str(args.pyshp_wheel.resolve()))
    import shapefile
    bounds=(-122.56,-122.15,47.47,47.77)
    vertices=[];codes=[];provenance=[]
    for path in args.water:
        with ZipFile(path) as archive:
            name,=[n for n in archive.namelist() if n.endswith('.shp')]
            with shapefile.Reader(shp=io.BytesIO(archive.read(name))) as reader:
                count=0
                for shape in reader.iterShapes():
                    left,bottom,right,top=shape.bbox
                    if right<bounds[0] or left>bounds[1] or top<bounds[2] or bottom>bounds[3]:continue
                    points=np.asarray(shape.points,dtype='float32')
                    markers=np.full(len(points),2,dtype='uint8');starts=np.asarray(shape.parts)
                    markers[starts]=1;markers[np.r_[starts[1:],len(points)]-1]=79
                    vertices.append(points);codes.append(markers);count+=1
            provenance.append(dict(archive=path.name,member=name,sha256=sha256(path.read_bytes()).hexdigest(),shapes=count))
    with np.load(args.streets,allow_pickle=False) as z:
        roads={name:z[name] for name in ('local_vertices','local_codes','major_vertices','major_codes')}
        road_metadata={name:z[name].tolist() for name in z.files if name not in roads}
    args.out.mkdir(parents=True,exist_ok=True)
    destination=args.out/'tiger2020_seattle.npz'
    np.savez_compressed(destination,water_vertices=np.concatenate(vertices),water_codes=np.concatenate(codes),**roads)
    metadata=dict(source='2020 U.S. Census Bureau TIGER/Line roads and area hydrography',
        water=provenance,streets_sha256=sha256(args.streets.read_bytes()).hexdigest(),streets_metadata=road_metadata,
        output_sha256=sha256(destination.read_bytes()).hexdigest(),contains_POI_data=False,bounds=bounds)
    (args.out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(status='complete',geometry_bytes=destination.stat().st_size,contains_POI_data=False)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--water',nargs='+',type=Path,required=True)
    p.add_argument('--streets',type=Path,required=True);p.add_argument('--pyshp-wheel',type=Path)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
