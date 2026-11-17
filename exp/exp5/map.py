"""Map completed attribution profiles with the existing public basemap geometry."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as DrawPath
from matplotlib.patches import PathPatch


def run(args):
    if args.map_inputs:
        protocol=json.loads((args.map_inputs/'complete.json').read_text());assert protocol['status']=='complete'
        input_file=args.map_inputs/'map_points.csv'
        assert sha256(input_file.read_bytes()).hexdigest()==protocol['points_sha256']
        attrs=pd.read_csv(input_file);regime=attrs.regime.to_numpy()
    else:
        if args.summary is None or args.bundle is None:raise ValueError('Provide map inputs or both summary and bundle')
        protocol=json.loads((args.summary/'summary_protocol.json').read_text())
        assert protocol['status'] in ('complete','profiles_complete_parent_pending')
        with ZipFile(args.bundle) as z:
            attrs=pd.read_csv(io.BytesIO(z.read('poi_attributes.csv')))
        input_file=args.summary/'private_consensus.npz'
        with np.load(input_file,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['poi_id'],attrs.placekey.to_numpy(dtype=str));regime=z['regime']
    with np.load(args.geometry,allow_pickle=False) as z:
        layers={name:DrawPath(z[name+'_vertices'],z[name+'_codes']) for name in ('water','local','major')}
    x=attrs.longitude.to_numpy();y=attrs.latitude.to_numpy()
    bounds=protocol['bounds'] if args.map_inputs else (-122.520,-122.185,47.490,47.764)
    shown=attrs.shown.to_numpy() if args.map_inputs else (x>=bounds[0])&(x<=bounds[1])&(y>=bounds[2])&(y<=bounds[3])
    names=['History-driven','Location-driven','Identity-driven','Parent-driven','Others-driven']
    colors=np.array(['#3B7C9C','#D98253','#639F91','#A383A7','#B8A35C'])
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
    fig=plt.figure(figsize=(10,5.2));grid=fig.add_gridspec(2,4,width_ratios=[1.8,1,1,1],
         left=.01,right=.99,top=.91,bottom=.04,wspace=.08,hspace=.17)
    overall=fig.add_subplot(grid[:,0]);facets=[fig.add_subplot(grid[0,i]) for i in range(1,4)]
    facets += [fig.add_subplot(grid[1,i]) for i in range(1,3)];legend=fig.add_subplot(grid[1,3])
    def background(ax):
        ax.set(xlim=bounds[:2],ylim=bounds[2:],xticks=[],yticks=[],facecolor='#F2F1EE')
        ax.set_aspect(1/np.cos(np.deg2rad(47.6)))
        for spine in ax.spines.values():spine.set_visible(False)
        ax.add_patch(PathPatch(layers['water'],facecolor='#D6E4EC',edgecolor='none',zorder=1))
        for name,width in [('local',.15),('major',.35)]:
            ax.add_patch(PathPatch(layers[name],facecolor='none',edgecolor='white',lw=width,zorder=2))
    background(overall)
    order=np.random.default_rng(20261114).permutation(np.flatnonzero(shown))
    order=np.concatenate([order[regime[order]<0],order[regime[order]>=0]])
    color=np.where((regime[order]>=0)[:,None],colors[np.maximum(regime[order],0),None],np.array([['#A6ABAF']])).ravel()
    overall.scatter(x[order],y[order],c=color,s=3.0,alpha=.8,lw=0,zorder=3)
    overall.set_title('All attribution regimes',loc='left',fontsize=12,fontweight='bold',pad=10)
    overall.text(-122.48,47.65,'Puget\nSound',fontsize=9,color='#587381',fontstyle='italic',ha='center',va='center')
    overall.text(-122.23,47.62,'Lake\nWashington',fontsize=9,color='#587381',fontstyle='italic',ha='center',va='center')
    scale_start=-122.48;scale_end=scale_start+5/(111.32*np.cos(np.deg2rad(47.6)))
    overall.plot([scale_start,scale_end],[47.502]*2,color='#233A49',lw=1)
    overall.text((scale_start+scale_end)/2,47.506,'5 km',fontsize=8,ha='center',color='#233A49')
    for k,ax in enumerate(facets):
        background(ax);selected=shown&(regime==k)
        ax.scatter(x[shown&~selected],y[shown&~selected],color='#B5BBC0',s=.7,alpha=.3,lw=0,zorder=3)
        ax.scatter(x[selected],y[selected],color=colors[k],s=2.0,alpha=.85,lw=0,zorder=4)
        ax.set_title(names[k],loc='left',fontsize=9.5,fontweight='bold',color=colors[k],pad=5)
    legend.set_axis_off();legend.set(xlim=(0,1),ylim=(0,1))
    legend.text(.04,.94,'Mapped POIs',fontsize=10.5,fontweight='bold')
    counts=[]
    for k,name in enumerate(names):
        count=protocol['regime_counts'][name] if args.map_inputs else int((shown&(regime==k)).sum())
        counts.append(count);yy=.79-k*.125
        legend.scatter(.06,yy,color=colors[k],s=24,lw=0)
        legend.text(.15,yy,name.replace('-driven',''),fontsize=9,va='center')
        legend.text(.98,yy,f'{count:,}',fontsize=9,ha='right',va='center')
    missing=protocol['mapped_unassigned'] if args.map_inputs else int((shown&(regime<0)).sum())
    legend.text(.04,.09,f'Unassigned: {missing:,}',fontsize=8.5,color='#58616A')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(args.out)
    fig.savefig(args.out,bbox_inches='tight',pad_inches=.04)
    fig.savefig(args.qa,bbox_inches='tight',pad_inches=.04,dpi=180);plt.close(fig)
    mapped=protocol['mapped_POIs'] if args.map_inputs else int(shown.sum())
    outside=protocol['outside_extent'] if args.map_inputs else int((~shown).sum())
    manifest=dict(period=protocol['period'],mapped_POIs=mapped,outside_extent=outside,
          mapped_unassigned=missing,regime_counts=dict(zip(names,counts)),
          attribution_sha256=sha256(input_file.read_bytes()).hexdigest(),
          geometry_sha256=sha256(args.geometry.read_bytes()).hexdigest(),figure_sha256=sha256(args.out.read_bytes()).hexdigest(),
          renderer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path=args.out.with_suffix('.json') if args.map_inputs else args.summary/'map_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='complete',mapped_POIs=mapped,unassigned=missing)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path)
    p.add_argument('--bundle',type=Path);p.add_argument('--map-inputs',type=Path)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--qa',type=Path,required=True)
    run(p.parse_args())
