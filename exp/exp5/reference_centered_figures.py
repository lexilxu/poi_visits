"""Render the two existing attribution figures with recalculated assignments."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
from exp.common.paths import DATA_ROOT, RESULTS_ROOT
from argparse import ArgumentParser
OUT=RESULTS_ROOT/'exp5/reference_centered'
import exp.common.compact_figures as plots
import exp.exp5.plot_map as maps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=OUT)
    p.add_argument('--cartography', type=Path, default=DATA_ROOT/'cartography')
    p.add_argument('--out', type=Path)
    args=p.parse_args()
    results=args.results
    assert json.loads((results/'complete.json').read_text())['status']=='complete'
    art=args.out or results/'art';art.mkdir(parents=True,exist_ok=True)
    assignment=pd.read_csv(results/'attribution_cluster_assignments.csv')
    groups=['History','Location','Identity','Parent','Others']
    labels=[g+'-driven' for g in groups]
    matrix=assignment.groupby('cluster')[[g+'_z' for g in groups]].mean().reindex(labels).to_numpy()
    counts=assignment.cluster.value_counts().reindex(labels).to_numpy()
    plots.style()
    # Keep the original fingerprint design, with space for readable labels.
    fig=plt.figure(figsize=(9.4,4.1))
    ax=fig.add_axes([.205,.245,.50,.64])
    limit=np.ceil(np.abs(matrix).max()*2)/2
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    cmap=LinearSegmentedColormap.from_list('centered_z',['#255D87','#F7F7F4','#B95321'])
    for row in range(5):
        for col in range(5):
            val=matrix[row,col]
            ax.add_patch(Rectangle((col+.015,row+.015),.97,.97,facecolor=cmap(norm(val)),edgecolor='none'))
            ax.text(col+.5,row+.5,f'{val:.2f}',ha='center',va='center',fontsize=12.8,
                    fontweight='bold' if row==col else 'normal',color='white' if abs(val)>limit*.65 else plots.INK)
    ax.set(xlim=(0,5),ylim=(5,0),xticks=np.arange(5)+.5,yticks=np.arange(5)+.5)
    ax.set_xticklabels(groups,weight='bold',fontsize=12)
    ax.set_yticklabels(labels,fontsize=12)
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=8)
    for spine in ax.spines.values():spine.set_visible(False)
    bx=fig.add_axes([.748,.245,.24,.64])
    bx.barh(np.arange(5)+.5,counts,height=.36,color='#7796AC')
    for y,n in enumerate(counts):bx.text(1,y+.5,f'{n:,}',transform=bx.get_yaxis_transform(),ha='right',va='center',fontsize=12)
    bx.set(xlim=(0,max(counts)*1.5),ylim=(5,0));bx.set_axis_off()
    bx.set_title('Regime size (POIs)',fontsize=12,weight='bold',loc='left',pad=12)
    cax=fig.add_axes([.205,.138,.50,.038])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=np.linspace(-limit,limit,5))
    cb.ax.tick_params(labelsize=10);cb.outline.set_linewidth(.5)
    fig.text(.455,.035,'Mean standardized attribution',ha='center',fontsize=11,weight='bold')
    plots.validate_text(fig,'Reference-centered attribution fingerprint')
    fig.savefig(art/'attribution_fingerprint.pdf')
    fig.savefig(art/'attribution_fingerprint.png',dpi=160)
    plt.close(fig)
    meta=pd.read_csv(results/'poi_metadata.csv')
    points=assignment.merge(meta,on='placekey',validate='one_to_one')
    xlo,xhi,ylo,yhi=maps.BOUNDS
    shown=points.loc[points.longitude.between(xlo,xhi)&points.latitude.between(ylo,yhi)].sort_values('placekey')
    codes=pd.Categorical(shown.cluster,categories=labels).codes
    sizes=np.bincount(codes,minlength=5)
    base=args.cartography
    fig=maps.draw_clusters(shown[['longitude','latitude']].to_numpy(),codes,labels,sizes,
                           [base/'king_county_areawater_2020.zip',base/'kitsap_county_areawater_2020.zip'],
                           maps.street_paths(base/'seattle_streets_tiger2020.npz'),named=True)
    plots.validate_text(fig,'Reference-centered attribution map')
    fig.savefig(art/'attribution_cluster_map.pdf',dpi=400)
    fig.savefig(art/'attribution_cluster_map.png',dpi=200)
    plt.close(fig)
    (results/'figure_audit.json').write_text(json.dumps(dict(mapped_POIs=len(shown),outside_extent=len(points)-len(shown),
             mapped_regime_counts=sizes.tolist(),text_bounds_and_overlap_check='passed'),indent=2))
    print('Rendered both attribution figures.',flush=True)


if __name__=='__main__':main()
