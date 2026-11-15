"""Publication figure from the audited AutoDL attribution-origin aggregates."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
source=root/'artifacts/revision_20261114/final_results/extension_20261114/attribution_origin'
shares=pd.read_csv(source/'group_shares.csv');transition=pd.read_csv(source/'regime_transition.csv',index_col=0)
audit=json.loads((source/'complete.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelweight':'bold','axes.titleweight':'bold',
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(9.4,3.9),gridspec_kw={'width_ratios':[1.05,1.15]})
labels=['History','Location','Identity','Parent','Other'];x=np.arange(5)
axes[0].bar(x-.18,shares.original_share*100,.34,label='Original origin',color='#315b83')
axes[0].bar(x+.18,shares.translated_share*100,.34,label='Missing-parent origin',color='#bf7035')
axes[0].set_xticks(x,labels,rotation=28,ha='right',fontweight='bold')
axes[0].set_ylabel('Mean attribution share (%)');axes[0].set_ylim(0,77)
axes[0].set_title('(a) Attribution shares',loc='left',pad=12)
axes[0].legend(frameon=False,fontsize=9,loc='upper right')
axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
values=transition.to_numpy();percent=100*values/values.sum(1,keepdims=True)
im=axes[1].imshow(percent,cmap='Blues',vmin=0,vmax=100,aspect='equal')
axes[1].set_xticks(x,labels,rotation=28,ha='right',fontweight='bold')
axes[1].set_yticks(x,labels,fontweight='bold')
axes[1].set_xlabel('Regime after translation');axes[1].set_ylabel('Original regime')
axes[1].set_title('(b) Regime transitions',loc='left',pad=12)
for i in range(5):
    for j in range(5):axes[1].text(j,i,f'{values[i,j]:,}',ha='center',va='center',fontsize=9,fontweight='bold',color='white' if percent[i,j]>55 else '#202a35')
cbar=fig.colorbar(im,ax=axes[1],fraction=.045,pad=.03);cbar.set_label('Share of original regime (%)',fontsize=9)
fig.subplots_adjust(left=.075,right=.94,bottom=.24,top=.87,wspace=.42)
fig.text(.5,.025,r'Paired forecasts unchanged: maximum $|\Delta\hat{Y}|=5.7\times10^{-14}$',ha='center',fontsize=11,fontweight='bold')
out=root/'paper/revision/attribution_origin_translation.pdf';fig.savefig(out,bbox_inches='tight')
fig.savefig(source/'attribution_origin_translation.png',dpi=180,bbox_inches='tight');plt.close(fig)
print(out)
