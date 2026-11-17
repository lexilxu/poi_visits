"""Render the original attribution fingerprint from audited remote summaries."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm,to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from exp.common.primary_figures import style, save, INK, GRAY


def main(args):
    record=json.loads((args.data/'complete.json').read_text())
    assert record['status']=='complete'
    file=args.data/'04_attribution_fingerprint.csv'
    frame=pd.read_csv(file)
    groups=['History','Location','Identity','Parent','Others']
    assert frame.display_label.tolist() in [
        [g+'-driven' for g in groups], [g+'-driven' for g in groups]+['Unassigned']]
    values=frame[groups].to_numpy();finite=values[np.isfinite(values)]
    limit=max(.5,np.ceil(np.abs(finite).max()*2)/2) if len(finite) else .5
    cmap=LinearSegmentedColormap.from_list('attribution_z',['#255D87','#F7F7F4','#B95321'])
    scalar=ScalarMappable(norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),cmap=cmap)
    style();fig=plt.figure(figsize=(10,4.8))
    ax=fig.add_axes([.22,.25,.50,.61]);counts=fig.add_axes([.76,.25,.22,.61])
    n=len(frame)
    for row in range(n):
        for col in range(5):
            value=values[row,col];available=np.isfinite(value)
            face=scalar.to_rgba(value) if available else '#E9EDF0'
            ax.add_patch(Rectangle((col+.014,row+.014),.972,.972,facecolor=face,lw=0))
            rgb=np.asarray(to_rgba(face)[:3]);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            color='white' if linear@np.array([.2126,.7152,.0722])<.179 else INK
            ax.text(col+.5,row+.5,f'{value:.2f}' if available else '-',ha='center',va='center',
                color=color,fontsize=12,weight='bold' if row==col else 'normal')
    ax.set(xlim=(0,5),ylim=(n,0));ax.set_yticks(np.arange(n)+.5,frame.display_label)
    ax.set_xticks(np.arange(5)+.5,groups);ax.xaxis.tick_top()
    ax.tick_params(length=0,pad=8,labelsize=11.5)
    for tick in ax.get_xticklabels():tick.set_weight('bold')
    for spine in ax.spines.values():spine.set_visible(False)
    centers=np.arange(n)+.5
    counts.barh(centers,frame.n_pois,height=.36,color='#7796AC',lw=0)
    counts.axvline(0,color='#CBD6DE',lw=.55)
    for y,count in zip(centers,frame.n_pois):
        counts.text(1,y,f'{count:,}',transform=counts.get_yaxis_transform(),ha='right',va='center',fontsize=11.5)
    counts.set(xlim=(0,max(1,frame.n_pois.max())*1.55),ylim=(n,0));counts.set_axis_off()
    counts.set_title('Regime size (POIs)',loc='left',fontsize=12,weight='bold',pad=11)
    cax=fig.add_axes([.22,.155,.50,.035]);cb=fig.colorbar(scalar,cax=cax,orientation='horizontal',ticks=np.linspace(-limit,limit,5))
    cb.ax.tick_params(labelsize=10,length=2.4,pad=3);cb.outline.set_linewidth(.45)
    cb.set_label('Mean standardized attribution',fontsize=12,weight='bold',labelpad=8)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(args.out)
    layout=save(fig,args.out)
    manifest=dict(status='rendered_pending_visual_review',input_sha256=sha256(file.read_bytes()).hexdigest(),
        checkpoint_sha256=record['checkpoint_sha256'],source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        output_sha256=sha256(args.out.read_bytes()).hexdigest(),layout_checks=layout,
        empty_cells='Unavailable standardized scores are shown as dashes; zero-sensitivity POIs remain Unassigned.')
    args.out.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=manifest['status'],rows=n)))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
