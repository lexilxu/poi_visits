"""Render the original paired-marker neighbor diagnostic, including empty sets."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
from exp.common.primary_figures import style, save, BLUE, RUST, GRAY, INK
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(args):
    record=json.loads((args.data/'complete.json').read_text());assert record['status']=='complete'
    path=args.data/'neighbor_effects.csv';frame=pd.read_csv(path)
    assert len(frame)==10 and frame.neighbor_group.is_unique
    rates=100*frame[['false_activations_removed','recoveries_lost']].to_numpy()
    finite=rates[np.isfinite(rates)];assert (finite>=0).all() and (finite<=100).all()
    endpoint=max(10,np.ceil(finite.max()/10)*10) if len(finite) else 10
    style();fig=plt.figure(figsize=(10.5,5.6));ax=fig.add_axes([.29,.18,.57,.65])
    y=np.arange(len(frame))
    for row,(left,right) in enumerate(rates):
        if np.isfinite(left) and np.isfinite(right):ax.plot([left,right],[row-.12,row+.12],color='#9AA7B1',lw=.85,zorder=1)
    first,=ax.plot(rates[:,0],y-.12,ls='none',marker='o',ms=6,color=BLUE,label='False activations removed')
    second,=ax.plot(rates[:,1],y+.12,ls='none',marker='D',ms=5.5,mfc='white',mec=RUST,mew=1,label='Recoveries lost')
    for row,pair in enumerate(rates):
        for column,value in enumerate(pair):
            if not np.isfinite(value):
                ax.text(1.035,row+(-.12 if column==0 else .12),'-',transform=ax.get_yaxis_transform(),
                    ha='center',va='center',fontsize=14,color=BLUE if column==0 else RUST,weight='bold')
    labels=frame.neighbor_group.str.removesuffix(' neighbors')
    ax.set_yticks(y,labels);ax.set(xlim=(-.025*endpoint,1.04*endpoint),ylim=(len(frame)-.6,-.6),xlabel='Share of reference events (%)')
    ax.set_xticks(np.arange(0,endpoint+1,20 if endpoint>=80 else 10))
    ax.axvline(0,lw=.55,color='#ABB5BD');ax.tick_params(axis='y',length=0,pad=9,labelsize=11)
    ax.tick_params(axis='x',length=3,pad=5);ax.spines['left'].set_visible(False)
    ax.axhline(7.5,lw=.6,color='#D9E2E9',ls='--')
    fig.text(.045,.90,'Neighbor group',fontsize=12,weight='bold')
    fig.legend(handles=[first,second],loc='upper left',bbox_to_anchor=(.28,.96),ncol=2,frameon=False,
        handletextpad=.5,columnspacing=1.8,fontsize=11)
    if (~np.isfinite(rates)).any():
        ax.text(1.035,-.9,'N/A',transform=ax.get_yaxis_transform(),ha='center',va='bottom',fontsize=10,weight='bold',color=GRAY)
        fig.text(.29,.035,'N/A: the corresponding reference set contains no events.',fontsize=10.5,color=GRAY)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(args.out)
    layout=save(fig,args.out)
    report=dict(status='rendered_pending_visual_review',input_sha256=sha256(path.read_bytes()).hexdigest(),
        checkpoint_sha256=record['checkpoint_sha256'],source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        output_sha256=sha256(args.out.read_bytes()).hexdigest(),layout_checks=layout,
        reference_events={key:frame[key].tolist() for key in ['beneficial_events','harmful_events','recovered_events']},
        caption_addition='The final two rows use matched-history replacement and an external-group control. Dashes mark empty reference sets.')
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],groups=len(frame))))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
