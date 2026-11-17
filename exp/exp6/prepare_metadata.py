"""Prepare sparse spatial graphs and pre-January metadata sensitivity inputs."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from exp.common.audit import load_panel


def prepare(args):
    raw,dates,attrs=load_panel(args.bundle)
    args.out.mkdir(parents=True,exist_ok=True)
    n=len(attrs);coords=np.deg2rad(attrs[['latitude','longitude']].to_numpy())
    xyz=np.column_stack((np.cos(coords[:,0])*np.cos(coords[:,1]),
                         np.cos(coords[:,0])*np.sin(coords[:,1]),np.sin(coords[:,0])))
    tree=cKDTree(xyz);rows=[];cols=[]
    # Exclude self explicitly, including for duplicate coordinates.
    _,nearest=tree.query(xyz,k=12)
    for j,candidates in enumerate(nearest):
        peers=candidates[candidates!=j][:10];rows.extend([j]*len(peers));cols.extend(peers.tolist())
    knn=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n));knn=knn.maximum(knn.T)
    indices=tree.query_ball_point(xyz,r=2*np.sin(1/6371.))
    rows=[];cols=[];weights=[]
    for j,candidates in enumerate(indices):
        peers=np.asarray(sorted(candidates));distance=2*np.arcsin(np.clip(np.linalg.norm(xyz[peers]-xyz[j],axis=1)/2,0,1))
        keep=peers!=j;rows.extend([j]*int(keep.sum()));cols.extend(peers[keep].tolist())
        weights.extend(np.exp(-np.square(distance[keep]*6371.)/2).tolist())
    rbf=sparse.csr_matrix((weights,(rows,cols)),shape=(n,n))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'archived_2020q4'))
    from exp.common.original_protocol import correlation_matrix
    correlation=correlation_matrix(raw[:50]);correlation.setdiag(0);correlation.eliminate_zeros()
    report=[]
    for name,graph in [('Distance KNN',knn),('Distance RBF',rbf),('Visit correlation',correlation)]:
        graph=sparse.diags(1/np.maximum(np.asarray(graph.sum(1)).ravel(),1e-12))@graph
        graph=graph.astype('float32').tocsr();graph.sort_indices()
        path=args.out/(name.replace(' ','_')+'.npz');sparse.save_npz(path,graph)
        report.append(dict(graph=name,edges=int(graph.nnz),sha256=sha256(path.read_bytes()).hexdigest(),
                           degree_mean=float(np.diff(graph.indptr).mean()),
                           propagation='Row-normalized neighbor weights, then add identity and normalize again'))
    candidate=pd.read_csv(args.attributes)
    if candidate.placekey.duplicated().any():raise ValueError('Duplicate source metadata rows')
    if not set(candidate.attribute_source_month.unique())<={'2020-11','2020-12'}:
        raise ValueError('Alternative metadata includes a later snapshot')
    crosswalk=candidate.set_index('safegraph_place_id').placekey.to_dict()
    candidate['parent_placekey']=candidate.parent_placekey.fillna(candidate.parent_safegraph_place_id.map(crosswalk))
    early=candidate.set_index('placekey').reindex(attrs.placekey).reset_index()
    missing=early.attribute_source_month.isna()
    # Missing coordinates use the pre-January observed median; categorical
    # attributes remain missing and follow the encoder's unknown-category rule.
    for column in ('latitude','longitude'):
        early[column]=early[column].fillna(early[column].median())
    early.to_csv(args.out/'earliest_metadata.csv',index=False)
    comparison=[]
    for column in ('latitude','longitude','naics_code','top_category','sub_category','brands','parent_placekey','open_hours'):
        old=attrs[column].fillna('').astype(str).to_numpy();new=early[column].fillna('').astype(str).to_numpy()
        comparison.append(dict(feature=column,different_among_matched=int(((old!=new)&~missing).sum())))
    metadata=dict(source_sha256=sha256(args.attributes.read_bytes()).hexdigest(),
                  prepared_sha256=sha256((args.out/'earliest_metadata.csv').read_bytes()).hexdigest(),
                  available_months=early.attribute_source_month.value_counts().to_dict(),
                  missing_metadata_records=int(missing.sum()),parent_pois=int(early.parent_placekey.notna().sum()),
                  original_parent_pois=int(attrs.parent_placekey.notna().sum()),
                  parent_crosswalk='Only identifiers co-observed in the 2020-11/2020-12 source records',
                  missing_rule='Median observed early coordinates; missing categorical inputs; missing opening hours',
                  comparison=comparison)
    args.report.write_text(json.dumps(dict(graphs=report,metadata=metadata),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(graphs=report,metadata=metadata)),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--attributes',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    prepare(p.parse_args())
