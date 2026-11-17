"""Produce original attribution table and figure inputs from corrected profiles."""
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,socket,sys
import numpy as np
import pandas as pd

GROUPS=['History','Location','Identity','Parent','Others']


def main(args):
    sys.path[:0]=[str(args.source),str(args.source/'exp/common'),str(args.source/'exp/common')]
    from exp.common.original_protocol import load_panel
    from exp.common.audit import EXPECTED_BUNDLE_SHA256
    attrs=load_panel(args.bundle,EXPECTED_BUNDLE_SHA256).attrs
    record=json.loads((args.profiles/'complete.json').read_text())
    frame=pd.read_csv(args.profiles/'poi_profiles.csv')
    np.testing.assert_array_equal(frame.placekey.to_numpy(dtype=str),attrs.placekey.to_numpy(dtype=str))
    shares=frame[[g+'_share' for g in GROUPS]].to_numpy()
    assigned=frame.total_own_attribution.to_numpy()>0
    np.testing.assert_allclose(shares[assigned].sum(1),1.)
    assert (frame.regime.eq('Unassigned').to_numpy()==~assigned).all()
    args.out.mkdir(parents=True,exist_ok=False)
    rows=[];fingerprints=[]
    labels=[g+'-driven' for g in GROUPS]+(['Unassigned'] if (~assigned).any() else [])
    for label in labels:
        ids=np.flatnonzero(frame.regime.eq(label).to_numpy());representatives=[]
        if len(ids):
            centroid=shares[ids].mean(0)
            if label!='Unassigned':
                representatives=ids[np.argsort(np.linalg.norm(shares[ids]-centroid,axis=1),kind='stable')[:3]]
            else:representatives=ids[:3]
        rows.append(dict(display_label=label,n_pois=len(ids),mae=float(frame.mae.iloc[ids].mean()) if len(ids) else np.nan,
            representative_pois='; '.join(attrs.location_name.iloc[representatives].astype(str)),
            **{g:float(frame[g+'_share'].iloc[ids].mean()) if len(ids) else np.nan for g in GROUPS}))
        fingerprints.append(dict(display_label=label,n_pois=len(ids),
            **{g:float(frame[g+'_z'].iloc[ids].mean()) if len(ids) and label!='Unassigned' else np.nan for g in GROUPS}))
    pd.DataFrame(rows).to_csv(args.out/'04_attribution_clusters.csv',index=False)
    pd.DataFrame(fingerprints).to_csv(args.out/'04_attribution_fingerprint.csv',index=False)
    category=pd.crosstab(attrs.category9,frame.regime).reindex(columns=labels,fill_value=0)
    category.insert(0,'n_pois',category.sum(1))
    category[labels]=category[labels].div(category.n_pois,axis=0)
    category['mae']=frame.groupby(attrs.category9).mae.mean()
    category.to_csv(args.out/'category_cluster_distribution.csv')
    parent=attrs.parent_placekey.fillna('').astype(str).str.strip().ne('').to_numpy()
    definitions=json.loads(args.parent_groups.read_text())
    parent_rows=[]
    for label,categories in list(definitions['groups'].items())+[('All embedded POIs',None)]:
        present=parent if categories is None else parent&attrs.top_category.isin(categories).to_numpy()
        selected=present&assigned
        share=float(shares[selected,3].mean()) if selected.any() else np.nan
        parent_rows.append(dict(group=label,n_pois=int(present.sum()),assigned_pois=int(selected.sum()),
            unassigned_pois=int((present&~assigned).sum()),parent_share=share,own_share=1-share,gap=2*share-1,
            mae=float(frame.mae[present].mean()) if present.any() else np.nan))
    pd.DataFrame(parent_rows).to_csv(args.out/'parent_attribution.csv',index=False)
    spatial=frame.copy()
    spatial['latitude']=attrs.latitude;spatial['longitude']=attrs.longitude;spatial['category9']=attrs.category9
    spatial.to_csv(args.out/'attribution_cluster_assignments.csv',index=False)
    report=dict(status=record['status'],profile_source=str(args.profiles),checkpoint_sha256=record['checkpoint_sha256'],
        profile_sha256=sha256((args.profiles/'poi_profiles.csv').read_bytes()).hexdigest(),
        parent_definitions_sha256=sha256(args.parent_groups.read_bytes()).hexdigest(),
        code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        pois=len(frame),unassigned_pois=int((~assigned).sum()),
        convention='Group summaries use the same corrected checkpoint. Attribution shares condition on positive total own-input sensitivity; all POIs remain in MAE and category counts.')
    (args.out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for name in ['source','bundle','profiles','parent-groups','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
