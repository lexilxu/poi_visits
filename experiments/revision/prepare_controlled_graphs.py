"""Build the requested k=5,10,20 matched graph controls without training."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit_protocol import load_panel, canonical_partition, EXPECTED_BUNDLE_SHA256
from controlled_checks import matched_neighbors


def run(bundle, out):
    _,_,attrs = load_panel(bundle)
    codes = pd.to_numeric(attrs.naics_code).astype('int64').astype(str)
    coarse, fine = canonical_partition(codes.str[:2]), canonical_partition(codes.str[:4])
    out.mkdir(parents=True,exist_ok=True)
    private = out/'private_graphs'; private.mkdir(exist_ok=True)
    rows = []
    for seed in (42,43,44,45,46):
        for k in (5,10,20):
            controls = matched_neighbors(coarse,fine,k,seed)
            degrees = None
            for name,peers in controls.items():
                degree = (peers >= 0).sum(1)
                if degrees is None:
                    degrees = degree
                np.testing.assert_array_equal(degree,degrees)
                for j, neighbors in enumerate(peers):
                    selected = neighbors[neighbors >= 0]
                    assert j not in selected and len(np.unique(selected)) == len(selected)
                digest = sha256(peers.astype('<i4').tobytes()).hexdigest()
                np.savez_compressed(private/f'{name}_k{k}_seed{seed}.npz',neighbors=peers,
                                    poi_id=attrs.placekey.to_numpy(dtype=str),sha256=digest)
                rows.append({'adjacency':name,'k':k,'seed':seed,'nodes':len(peers),
                             'mean_degree':float(degree.mean()),'median_degree':float(np.median(degree)),
                             'isolated_pois':int((degree == 0).sum()),
                             'candidate_shortfall_pois':int((degree < k).sum()),
                             'identical_row_degrees_across_controls':True,'edge_index_sha256':digest})
    pd.DataFrame(rows).to_csv(out/'controlled_graph_statistics.csv',index=False)
    (out/'controlled_graph_protocol.json').write_text(json.dumps({
        'bundle_sha256':EXPECTED_BUNDLE_SHA256,'seeds':[42,43,44,45,46], 'k':[5,10,20],
        'degree_rule':'d_j=min(k, number of other NAICS-4 members); same d_j in all three graphs',
        'sampling':'Uniform without replacement, no self neighbors; labels fixed before outcomes',
        'operator':'0.5 own + 0.5 mean peers; isolated rows use own only',
        'status':'Graphs built and validated; predictive comparisons not yet trained.'
    },indent=2)+'\n',encoding='utf8')
    print(pd.DataFrame(rows).query('seed == 42').to_string(index=False))


if __name__ == '__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); run(args.bundle,args.out)
