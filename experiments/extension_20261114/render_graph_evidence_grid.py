"""Render the audited evidence grid in the reference manuscript figure style."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[2]


def labels(cuts):
    def number(value):
        return f'{value:.2f}'.rstrip('0').rstrip('.')
    return [('0' if x == 0 else '\u2264'+number(x)) for x in cuts]+['>'+number(cuts[-1])]


def render(source, destination, preview):
    table=pd.read_csv(source/'cells.csv');audit=json.loads((source/'audit.json').read_text())
    assert audit['status']=='complete'
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':10.8,
                         'axes.labelweight':'bold','axes.titleweight':'bold',
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':True})
    cmap=LinearSegmentedColormap.from_list('evidence_gain',
          ['#8F2D26','#BD6B4B','#E3B49A','#F7F6F2','#BAD2DC','#6296AD','#285A80'])
    limit=max(.05,np.ceil(table.loc[table.display_cell,'MAE_gain_mean'].abs().max()*20)/20)
    scale=ScalarMappable(norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),cmap=cmap)
    fig=plt.figure(figsize=(7.3,7.35))
    axes=[fig.add_axes([x,y,.325,.323]) for y in [.588,.145] for x in [.10,.53]]
    supported=[]
    for k,(ax,(period,graph)) in enumerate(zip(axes,[(p,g) for p in ['december','january'] for g in ['NAICS-2','NAICS-4']])):
        panel=next(r for r in audit['panels'] if r['period']==period and r['adjacency']==graph)
        rows=table[(table.period==period)&(table.adjacency==graph)]
        ny,nx=panel['grid_shape']
        for row in rows.itertuples():
            color=scale.to_rgba(row.MAE_gain_mean) if row.display_cell else '#EBEEF0'
            ax.add_patch(Rectangle((row.x_bin+.025,row.y_bin+.025),.95,.95,
                                  facecolor=color,edgecolor='none'))
            if row.display_cell:
                rgb=np.array(color[:3]);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                ink='white' if linear@np.array([.2126,.7152,.0722])<.19 else '#17232C'
                value='0.000' if abs(row.MAE_gain_mean)<.0005 else f'{row.MAE_gain_mean:+.3f}'
                ax.text(row.x_bin+.5,row.y_bin+.5,value,ha='center',va='center',
                        fontsize=8.7,fontweight='bold',color=ink)
                supported.append(row.events)
            else:
                ax.text(row.x_bin+.5,row.y_bin+.5,'\u2014',ha='center',va='center',fontsize=10,color='#68747D')
        ax.set(xlim=(0,nx),ylim=(0,ny),aspect='equal')
        ax.set_xticks(np.arange(nx)+.5,labels(panel['x_cutpoints']))
        ax.set_yticks(np.arange(ny)+.5,labels(panel['y_cutpoints']))
        ax.tick_params(length=0,pad=4,labelsize=8.5)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontweight('bold')
        for spine in ax.spines.values():spine.set_visible(False)
        if k>=2:ax.set_xlabel('Target evidence (visits/day)',labelpad=8)
        if k%2==0:ax.set_ylabel('Neighbor evidence (visits/day)',labelpad=8)
        pos=ax.get_position()
        month='December 2020' if period=='december' else 'January 2021'
        fig.text(pos.x0,pos.y1+.043,f'({"abcd"[k]}) {month}',fontsize=11,fontweight='bold')
        fig.text(pos.x0,pos.y1+.019,graph,fontsize=10.5,fontweight='bold',color='#285A80')
        fig.text(pos.x1,pos.y1+.019,f'Mean {panel["overall_gain"]:+.3f}',ha='right',fontsize=9.2,
                 color='#96492F' if panel['overall_gain']<0 else '#285A80')
    cax=fig.add_axes([.897,.33,.018,.38]);bar=fig.colorbar(scale,cax=cax)
    bar.set_label('MAE gain (visits)',fontsize=10,fontweight='bold',labelpad=7)
    bar.set_ticks(np.linspace(-limit,limit,5));bar.ax.tick_params(labelsize=8.5,length=2.5,pad=3)
    bar.outline.set_linewidth(.5)
    counts={p:next(r['events'] for r in audit['panels'] if r['period']==p) for p in ['december','january']}
    fig.text(.48,.065,f'Scored events: December {counts["december"]:,}  |  January {counts["january"]:,}',ha='center',fontsize=8.7,color='#4E5962')
    fig.text(.48,.039,'Five-seed means  |  Positive gain = lower MAE',ha='center',fontsize=10,fontweight='bold')
    fig.text(.48,.014,'Validation-defined bins; tick labels give upper bounds and the final open tail.',ha='center',fontsize=8.7,color='#4E5962')
    destination.parent.mkdir(parents=True,exist_ok=True);preview.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,bbox_inches='tight',pad_inches=.035)
    fig.savefig(preview,bbox_inches='tight',pad_inches=.035,dpi=200);plt.close(fig)
    receipt=dict(source_sha256={n:sha256((source/n).read_bytes()).hexdigest() for n in ['cells.csv','audit.json']},
                 pdf_sha256=sha256(destination.read_bytes()).hexdigest(),color_limit=limit,
                 minimum_displayed_events=min(supported),cells=len(table),displayed_cells=int(table.display_cell.sum()))
    (preview.parent/'graph_figure_render.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    render(ROOT/'artifacts/revision_20261114/final_results/extension_20261114/graph_evidence_grid',
           ROOT/'paper/revision/art/pre_outcome_graph_transfer.pdf',
           ROOT/'artifacts/revision_20261114/private_substantive_20261114/graph_figure/preview.png')
