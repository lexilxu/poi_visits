"""Render corrected primary figures in the original reference-paper style."""
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties,findfont
import numpy as np
import pandas as pd

BLUE='#285A80';TEAL='#21847D';RUST='#B45D3B';PLUM='#745483';GRAY='#74808A';INK='#222A31'


def style():
    try:findfont(FontProperties(family='Arial'),fallback_to_default=False);font='Arial'
    except ValueError:font='DejaVu Sans'
    plt.rcParams.update({'font.family':font,'font.size':13,'axes.labelsize':14,'axes.labelweight':'bold',
        'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'axes.edgecolor':INK,'text.color':INK,
        'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'figure.facecolor':'white',
        'savefig.facecolor':'white','pdf.fonttype':42,'figure.dpi':160})


def layout_report(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();records=[];width,height=fig.canvas.get_width_height()
    omitted=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lower,upper=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<lower-1e-9 or tick.get_loc()>upper+1e-9:
                    omitted.update([id(tick.label1),id(tick.label2)])
    for artist in fig.findobj(Text):
        if id(artist) not in omitted and artist.get_visible() and artist.get_text().strip():
            box=Text.get_window_extent(artist,renderer)
            if box.width>0 and box.height>0:records.append((artist.get_text(),box))
    outside=[text for text,b in records if min(b.x0,b.y0)<-1 or b.x1>width+1 or b.y1>height+1]
    overlaps=[]
    for i,(text,a) in enumerate(records):
        for other,b in records[i+1:]:
            overlap=Bbox.intersection(a,b)
            if overlap is not None and min(overlap.width,overlap.height)>2.5:overlaps.append([text,other])
    return dict(text_items=len(records),outside=outside,text_overlaps=overlaps)


def save(fig,path):
    report=layout_report(fig)
    fig.savefig(path.with_suffix('.png'),dpi=180)
    if report['outside'] or report['text_overlaps']:
        path.with_suffix('.layout.json').write_text(json.dumps(report,indent=2))
        plt.close(fig);raise RuntimeError(json.dumps(dict(figure=path.name,**report)))
    fig.savefig(path,metadata={'Title':path.stem.replace('_',' '),'Subject':'Corrected EvolveGCN-H primary-seed results'})
    plt.close(fig);return report


def horizons(frame,path):
    fig,ax=plt.subplots(figsize=(9.3,4.8));fig.subplots_adjust(left=.10,right=.98,top=.97,bottom=.31)
    styles=[('EvolveGCN (identity)',BLUE,'o','-',True),
        ('EvolveGCN (best non-identity)',BLUE,'s','--',False),('MTGNN (best)',TEAL,'^','-.',True),
        ('Historical average',GRAY,None,':',True),('AR-ridge',RUST,'v','-',True),('LSTM',PLUM,'D','-',False)]
    for label,color,marker,line,filled in styles:
        values=frame[frame.series.eq(label)].sort_values('H');assert values.H.tolist()==[7,14,28]
        drawn,=ax.plot(values.H,values.skill_percent,label=label,color=color,marker=marker,linestyle=line,
            linewidth=1.5,markersize=5.5,markeredgewidth=.85,markerfacecolor=color if filled else 'white')
        np.testing.assert_array_equal(drawn.get_ydata(),values.skill_percent)
    low=float(frame.skill_percent.min());high=float(frame.skill_percent.max())
    ax.set(xlim=(5.8,29.2),ylim=(10*np.floor((low-3)/10),10*np.ceil((high+3)/10)),
        xlabel=r'Forecast horizon $H$ (days)',ylabel='MAE skill (%)')
    ax.set_xticks([7,14,28]);ax.grid(axis='y',color='#D9E2E9',linewidth=.6);ax.set_axisbelow(True)
    ax.tick_params(length=3,pad=5);ax.xaxis.labelpad=8
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.015),ncol=3,
        frameon=False,handlelength=2.8,columnspacing=1.8,labelspacing=1.1)
    return save(fig,path)


def annotate(ax,point,position,text,color=INK,ha='left',size=10.5):
    return ax.annotate(text,xy=point,xytext=position,ha=ha,va='center',fontsize=size,color=color,
        arrowprops=dict(arrowstyle='-',color='#A2ABB2',lw=.65,shrinkA=3,shrinkB=4))


def recovery(frame,path):
    fig=plt.figure(figsize=(10.2,6.4));ax=fig.add_axes([.10,.115,.87,.76])
    styles={'EvolveGCN':dict(marker='o',color=BLUE,mfc=BLUE),
        'MTGNN':dict(marker='^',color=TEAL,mfc=TEAL),'Historical average':dict(marker='+',color=GRAY,mfc=GRAY),
        'AR-ridge':dict(marker='v',color=RUST,mfc=RUST),'LSTM':dict(marker='D',color=PLUM,mfc='white')}
    ymax=25*np.ceil((100*frame.dmr_model.max()+12)/25)
    ax.set(xlim=(-3,105),ylim=(0,ymax),xlabel='False activation rate (%)',ylabel='Demand mass recovery ratio (%)')
    ax.set_xticks([0,20,40,60,80,100]);ax.set_yticks(np.arange(0,ymax+1,25));ax.tick_params(length=3,pad=5)
    ax.grid(axis='y',color='#E0E6EB',lw=.5);ax.set_axisbelow(True)
    ordered=frame[frame.model.eq('EvolveGCN')].sort_values('dmr_model')
    label_y=100*ordered.dmr_model.to_numpy().copy()
    gap=.065*ymax
    for index in range(1,len(label_y)):
        label_y[index]=max(label_y[index],label_y[index-1]+gap)
    if label_y[-1]>.94*ymax:label_y-=label_y[-1]-.94*ymax
    positions={name:(82,y) for name,y in zip(ordered.adjacency,label_y)}
    for row in frame[frame.model.eq('EvolveGCN')].itertuples():
        point=(100*row.far_0_5,100*row.dmr_model)
        ax.plot(*point,ls='none',ms=6,mew=.8,**styles[row.model])
        annotate(ax,point,positions[row.adjacency],row.adjacency,BLUE,ha='right',size=13)
    inset=fig.add_axes([.18,.48,.37,.31],facecolor='white')
    inset.set(xlim=(-.10,2.3),ylim=(5.4,23.8))
    inset.set_xticks([0,1,2]);inset.set_yticks([5,10,15,20]);inset.tick_params(labelsize=11.5,length=2.5,pad=3)
    inset.set_title('Low-activation detail',fontsize=13.5,weight='bold',loc='left',pad=10)
    inset_positions={('Historical average','None'):(.02,22.1),('LSTM','None'):(.50,20.5),('AR-ridge','None'):(1.05,18.3),
        ('MTGNN','Identity'):(.02,6.4),('MTGNN','Top category'):(.02,9.0),
        ('MTGNN','NAICS-2'):(.02,11.6),('MTGNN','Distance KNN'):(1.05,16.2)}
    detail=frame[~frame.model.eq('EvolveGCN')]
    assert ((100*detail.far_0_5<2.3)&(100*detail.dmr_model<23.8)).all()
    ax.plot(100*detail.far_0_5,100*detail.dmr_model,ls='none',marker='.',ms=2,color=GRAY)
    for row in detail.itertuples():
        point=(100*row.far_0_5,100*row.dmr_model)
        inset.plot(*point,ls='none',ms=5.3,mew=.8,**styles[row.model])
        name=row.adjacency if row.model=='MTGNN' else row.model
        annotate(inset,point,inset_positions[(row.model,row.adjacency)],name,styles[row.model]['color'],size=12)
    ax.add_patch(Rectangle((0,5.4),2.3,18.4,fill=False,lw=.8,edgecolor=GRAY))
    ax.annotate('Expanded above',xy=(2.3,23.8),xytext=(8,40),fontsize=12,color=GRAY,
        arrowprops=dict(arrowstyle='-',color=GRAY,lw=.7))
    handles=[Line2D([],[],ls='none',ms=6,label=name,**style) for name,style in styles.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.98),ncol=5,frameon=False,
        handletextpad=.4,columnspacing=1.6)
    return save(fig,path)


def outcomes(frame,meta,palette,path):
    fig=plt.figure(figsize=(10.5,5.3));axes=[fig.add_axes([x,.19,.345,.64]) for x in [.095,.505]]
    values=frame.loc[frame.display_cell,'mean_mae_gain'].dropna().to_numpy()
    limit=max(.05,np.ceil(np.abs(values).max()*20)/20)
    cmap=ListedColormap(np.loadtxt(palette),name='vik').reversed().with_extremes(bad='#E9EDF0')
    scalar=ScalarMappable(norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),cmap=cmap)
    for panel,(axis,summary) in enumerate(zip(axes,meta['outcome_summary'])):
        rows=frame[frame.outcome.eq(summary['outcome'])]
        nx,ny=int(rows.x_bin.max())+1,int(rows.y_bin.max())+1
        for row in rows.itertuples():
            value=row.mean_mae_gain if row.display_cell else np.nan
            face=scalar.to_rgba(value) if np.isfinite(value) else '#E9EDF0'
            axis.add_patch(Rectangle((row.x_bin+.025,row.y_bin+.025),.95,.95,facecolor=face,lw=0))
            if np.isfinite(value):
                rgb=np.asarray(face[:3]);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                color='white' if linear@np.array([.2126,.7152,.0722])<.179 else INK
                label=f'{value:+.2f}' if abs(value)>=.005 else '0.00'
            else:color=GRAY;label='-'
            axis.text(row.x_bin+.5,row.y_bin+.5,label,ha='center',va='center',fontsize=14,weight='bold',color=color)
        axis.set(xlim=(0,nx),ylim=(0,ny))
        axis.set_xticks(np.arange(nx+1),[f'{x:.2f}'.rstrip('0').rstrip('.') for x in meta['target_edges']])
        axis.set_yticks(np.arange(ny+1),[f'{x:.2f}'.rstrip('0').rstrip('.') for x in meta['neighbor_edges']])
        axis.tick_params(length=2.5,pad=4,labelsize=12)
        for spine in axis.spines.values():spine.set_visible(False)
        axis.set_xlabel('Target evidence (visits/day)',labelpad=10)
        if panel==0:axis.set_ylabel('Neighbor evidence (visits/day)',labelpad=10)
        else:axis.set_yticklabels([])
        axis.set_title(('a  ' if panel==0 else 'b  ')+summary['outcome'],loc='left',fontsize=14,weight='bold',pad=28)
        axis.text(1,1.025,f"Mean {summary['mean_mae_gain']:+.3f}",transform=axis.transAxes,
            ha='right',va='bottom',fontsize=12.5,color=BLUE if summary['mean_mae_gain']>=0 else RUST)
    cax=fig.add_axes([.89,.19,.017,.64]);cb=fig.colorbar(scalar,cax=cax)
    cb.set_label('MAE gain (visits)',fontsize=14,weight='bold',labelpad=12);cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(.5)
    return save(fig,path)


def main(args):
    metadata=json.loads((args.data/'complete.json').read_text());assert metadata['status']=='complete_primary_figure_inputs'
    args.out.mkdir(parents=True,exist_ok=False);style()
    frame=pd.read_csv(args.data/'07_horizon_skill.csv',keep_default_na=False)
    reports={'forecasting_performance_across_horizons.pdf':horizons(frame,args.out/'forecasting_performance_across_horizons.pdf')}
    frame=pd.read_csv(args.data/'01_recovery_activation.csv',keep_default_na=False)
    reports['recovery_activation_tradeoff.pdf']=recovery(frame,args.out/'recovery_activation_tradeoff.pdf')
    frame=pd.read_csv(args.data/'02_graph_transfer_cells.csv')
    reports['outcome_conditioned_graph_transfer.pdf']=outcomes(frame,metadata,args.palette,args.out/'outcome_conditioned_graph_transfer.pdf')
    report=dict(status='rendered_pending_visual_review',seed=metadata['seed'],layout_checks=reports,
        inputs={p.name:sha256(p.read_bytes()).hexdigest() for p in args.data.iterdir() if p.suffix in ['.json','.csv']},
        palette_sha256=sha256(args.palette.read_bytes()).hexdigest(),code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={p.name:sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*.pdf')},
        recovery_caption_addition='The inset expands the low-activation configurations shown by the outlined region.')
    (args.out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],figures=len(reports))))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for field in ['data','out','palette']:p.add_argument('--'+field,type=Path,required=True)
    main(p.parse_args())
