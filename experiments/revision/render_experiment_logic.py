"""Draw the experiment logic with the manuscript's section titles."""
from pathlib import Path
from hashlib import sha256
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper/revision'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(9.4,4.9));fig.subplots_adjust(left=.01,right=.99,bottom=.025,top=.99)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ink='#233A49';blue='#3B7C9C';teal='#639F91';rust='#D98253'


def box(x,y,w,h,color,title,lines):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='none'))
    ax.text(x+.015,y+h-.065,title,fontsize=13,fontweight='bold',color=ink,va='top',linespacing=1.18)
    for i,line in enumerate(lines):
        ax.text(x+.015,y+h-.195-i*.094,line,fontsize=11.3,color=ink,va='top',linespacing=1.15)


ax.add_patch(Rectangle((.025,.805),.95,.17,facecolor='#F0F2F3',edgecolor='none'))
ax.text(.045,.94,'Parameter Settings',fontsize=13.5,fontweight='bold',color=ink,va='center')
ax.text(.045,.885,'Complete seven-day forecasts  ·  Two input lengths  ·  Five fitted seeds',fontsize=12,color=ink,va='center')
ax.text(.045,.836,'Validation-only selection  ·  December 2020 test  ·  January 2021 evaluation',fontsize=12,color=ink,va='center')
ax.text(.025,.737,'Prediction Task',fontsize=14,fontweight='bold',color=blue,va='center')
ax.text(.335,.737,'Attribution Task',fontsize=14,fontweight='bold',color=teal,va='center')
for left,right,color in [(.025,.315,blue),(.335,.975,teal)]:
    ax.plot([left,right],[.704,.704],color=color,lw=1.1)
box(.025,.185,.29,.49,'#EAF1F5','Forecasting\nPerformance',[
    'Input Windows and\nForecast Leads','Low-History POIs'])
box(.335,.185,.29,.49,'#EAF3EF','Information Sharing\namong POIs',[
    'Alternative\nAdjacency Matrices','Related-POI\nPerturbation'])
box(.665,.185,.31,.49,'#EAF3EF','Explaining POI\nVisit Predictions',[
    'POI Attribution Regimes','Parent-Place Information','Category and\nIndustry Granularity'])
for x in (.17,.48,.82):
    ax.add_patch(FancyArrowPatch((x,.805),(x,.76),arrowstyle='-|>',mutation_scale=12,lw=1,color=ink))
ax.add_patch(FancyArrowPatch((.625,.43),(.662,.43),arrowstyle='-|>',mutation_scale=12,lw=1,color=ink))
ax.text(.025,.11,'Paired temporal and spatial uncertainty',fontsize=12.2,fontweight='bold',color=ink)
ax.text(.025,.052,'Loss and calibration  ·  Matched graph controls  ·  Retraining and attribution stability',fontsize=11.8,color=ink)
out=PAPER/'art/exp_logic.pdf';fig.savefig(out,bbox_inches='tight',pad_inches=.04)
qa=ROOT/'artifacts/revision_20261114/qa';qa.mkdir(exist_ok=True)
fig.savefig(qa/'exp_logic.png',dpi=180,bbox_inches='tight',pad_inches=.04);plt.close(fig)
(ROOT/'artifacts/revision_20261114/experiment_logic_manifest.json').write_text(json.dumps(dict(
    source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    figure_sha256=sha256(out.read_bytes()).hexdigest(),
    scope='Titles correspond to the existing sections and subsections; no experiment result is encoded.'),indent=2)+'\n')
print('Rendered experiment logic from the existing manuscript titles.')
