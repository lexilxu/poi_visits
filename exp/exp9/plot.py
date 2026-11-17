"""Format verified remote aggregates into additions-only Markdown and a paper figure."""
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from exp.common.paths import DATA_ROOT, RESULTS_ROOT, REPO_ROOT


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=RESULTS_ROOT/'exp9')
    args = parser.parse_args()

    ROOT=Path(__file__).resolve().parents[2]
    RESULTS=args.results
    OUT=args.out
    OUT.mkdir(parents=True,exist_ok=True)
    EVIDENCE=OUT/'evidence';EVIDENCE.mkdir(exist_ok=True)
    for f in RESULTS.iterdir():shutil.copy2(f,EVIDENCE/f.name)
    for name in ['analyze.py','verify.py','PROTOCOL.md']:
        shutil.copy2(Path(__file__).with_name(name),EVIDENCE/name)
    gain=pd.read_csv(RESULTS/'strata_gain_intervals.csv')
    cost=pd.read_csv(RESULTS/'cost_summary.csv')
    strata=pd.read_csv(RESULTS/'strata_summary.csv')
    recovery=pd.read_csv(RESULTS/'recovery_summary.csv')
    pair=pd.read_csv(RESULTS/'cost_pair_intervals.csv')
    models=['Historical average','AR-ridge','Shared LSTM','Matched LSTM','MTGNN (NAICS-2)','GraphGRU (Identity)','GraphGRU (NAICS-2)','GraphGRU (NAICS-4)']
    cells=['Low/Low','Low/High','High/Low','High/High']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,'axes.labelweight':'bold','axes.titleweight':'bold','axes.titlesize':13,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.8),sharex=True,sharey=True)
    for row,graph in enumerate(['NAICS-2','NAICS-4']):
        for col,period in enumerate(['December','January']):
            ax=axes[row,col]
            for k,(model,color,marker,offset) in enumerate([('GraphGRU (NAICS-2)','#176B86','o',-.12),('GraphGRU (NAICS-4)','#B55B32','s',.12)]):
                d=gain[(gain.period==period)&(gain.neighbor_partition==graph)&(gain.model==model)].set_index('cell').loc[cells]
                x=d.estimate.to_numpy();lo=d.ci_low.to_numpy();hi=d.ci_high.to_numpy()
                ax.errorbar(x,np.arange(4)+offset,xerr=np.array([x-lo,hi-x]),fmt=marker,color=color,markersize=7,capsize=3,elinewidth=1.7,label=model)
            ax.axvline(0,color='#333333',linewidth=1,linestyle='--')
            ax.set_yticks(range(4),['Low / Low','Low / High','High / Low','High / High'])
            ax.set_title(f'({chr(97+row*2+col)}) {period} | {graph} neighbors',pad=11)
            ax.set_xlim(-.205,.025);ax.set_ylim(3.5,-.5)
            ax.set_xticks([-.20,-.15,-.10,-.05,0])
            ax.grid(axis='x',color='#E5E5E5',linewidth=.7)
            ax.set_axisbelow(True)
            if col==0:ax.set_ylabel('Target / neighbor activity',labelpad=11)
            if row==1:ax.set_xlabel('MAE gain over Identity (visits)',labelpad=10)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.995),ncol=2,frameon=False,fontsize=13)
    fig.subplots_adjust(left=.16,right=.98,top=.88,bottom=.11,hspace=.35,wspace=.14)
    fig.savefig(OUT/'forecast_time_transfer.pdf',bbox_inches='tight')
    fig.savefig(OUT/'forecast_time_transfer.png',dpi=190,bbox_inches='tight')
    plt.close(fig)

    matched=r'''The complete-window comparison in Appendix~A.1 evaluates five seeds at \((T,H)=(14,7)\), followed by January evaluation with frozen fitted models. MTGNN (NAICS-2) achieves MAE of \(1.222\pm0.008\) in December and \(1.244\pm0.011\) in January. Matched LSTM and GraphGRU (Identity) are close: their respective MAEs are \(1.298\pm0.015\) and \(1.302\pm0.007\) in December, and \(1.309\pm0.014\) and \(1.311\pm0.007\) in January, where \(\pm\) denotes the standard deviation across seeds. Paired intervals for the matched-model MAE differences contain zero in both months. These results support the MTGNN accuracy advantage across the two months and comparable accuracy of the matched recurrent specifications. The \(H=14\) and \(H=28\) results provide exploratory horizon comparisons.'''

    transfer=r'''\paragraph{Forecast-time history strata.}
    We evaluate whether target and neighbor evidence available at the forecast origin identifies useful adjacency configurations. This analysis uses the complete-window \((T,H)=(14,7)\) protocol in Appendix~A.1 and the existing five-seed EvolveGCN predictions. Target and neighbor evidence follow Section~4.4, measured in observed-visit units over the preceding 14 days; missing history entries retain the zero placeholders used by the models. The analysis covers 9,866 POIs with non-self neighbors under both NAICS-2 and NAICS-4. Low and high activity are separated by medians pooled across these POIs and the 30 training origins. The fixed cutoffs are 1.3571 visits for target activity, 2.9191 for NAICS-2 neighbor activity, and 2.9497 for NAICS-4 neighbor activity; ties enter the low group. Each neighbor definition yields four strata, fixed at the forecast origin for all seven leads. Within each stratum, Identity, NAICS-2, and NAICS-4 are compared on identical observed events. The cutoffs are carried unchanged into January. Intervals use 2,000 paired bootstrap replicates combining seven-origin blocks and 3~km spatial cells, after averaging event losses across the five seeds.

    Identity has the lowest mean MAE in every stratum under both neighbor definitions in December and January (Figure~\ref{fig:forecast_time_transfer}). For low target and high neighbor activity, the MAE gain of NAICS-2 under its corresponding partition is \(-0.046\) in December and \(-0.112\) in January; the corresponding NAICS-4 gains are \(-0.110\) and \(-0.129\). In the NAICS-4 low/high stratum, NAICS-4 raises DMR from 28.9\% to 33.2\% in December and from 26.9\% to 34.8\% in January, while FAR rises from 22.0\% to 26.6\% and from 21.3\% to 28.3\%, respectively. Thus, greater recovered visit volume can accompany higher overall error. These forecast-time comparisons support self-history as the default EvolveGCN adjacency for MAE in the evaluated cohort and periods.

    \begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{forecast_time_transfer.pdf}
    \caption{Forecast-time stratification using training-period activity cutoffs. Points show five-seed mean MAE gain relative to GraphGRU (Identity); positive values favor the displayed non-identity adjacency. Bars show pointwise 95\% paired intervals conditional on the five fitted seeds. Each panel uses a common event partition for both adjacency comparisons. The intervals lie below zero in 31 of the 32 comparisons; the December NAICS-4 comparison in the low/low stratum contains zero.}
    \label{fig:forecast_time_transfer}
    \end{figure}'''

    cost_method=r'''\paragraph{Asymmetric forecast preferences.}
    To quantify the recovery--false activation tradeoff, we evaluate the existing complete-window forecasts under asymmetric absolute loss,
    \begin{equation}
    L_{\lambda}=\frac{1}{|\mathcal{E}|}
    \sum_{(b,h,j)\in\mathcal{E}}
    \left[\lambda\bigl(Y_{b,h,j}-\widehat{Y}_{b,h,j}\bigr)_{+}
    +\bigl(\widehat{Y}_{b,h,j}-Y_{b,h,j}\bigr)_{+}\right],
    \label{eq:asymmetric_visit_loss}
    \end{equation}
    where \(\mathcal{E}\) is the set of observed forecast events in the evaluated cohort, \((z)_{+}=\max\{z,0\}\), and \(\lambda\) weights underprediction relative to overprediction. Predictions are evaluated in nonnegative visit units. At \(\lambda=1\), the loss equals MAE; larger \(\lambda\) assigns greater cost to missed visit volume. We evaluate \(\lambda\in\{0.5,1,2,3,5\}\) on all POIs and on the 1,944 low-history POIs identified from the 50 training days of the complete-window protocol. The weights represent forecasting preferences for sampled observed visits.

    Table~\ref{tab:asymmetric_low_history} shows a change in the low-history ranking. MTGNN has the lowest mean loss at \(\lambda=0.5\) and \(\lambda=1\) in both months. At \(\lambda=5\), Shared LSTM has the lowest mean loss in December and GraphGRU (Identity) in January. The MTGNN-minus-Identity loss difference changes from \(-0.0193\) to \(+0.0266\) in December and from \(-0.0153\) to \(+0.0531\) in January as \(\lambda\) increases from 1 to 5. At \(\lambda=5\), the paired 95\% intervals are \([0.0208,0.0332]\) and \([0.0148,0.1097]\), respectively. Across all 9,884 POIs, MTGNN has the lowest mean loss at \(\lambda=0.5,1\), whereas the historical average has the lowest mean loss at \(\lambda=3,5\) in both months. Forecast preferences therefore affect model rankings differently in the sparse cohort and the full panel.'''

    cost_table=r'''\begin{table}[htbp]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \caption{Asymmetric loss on the 1,944 low-history POIs under the complete-window protocol. Neural entries average losses across five fitted seeds; deterministic baselines use one forecast. Bold values mark the lowest mean in each month and weight column.}
    \label{tab:asymmetric_low_history}
    \begin{tabular}{lrrrrr}
    \toprule
    Model & $\lambda=0.5$ & $1$ & $2$ & $3$ & $5$ \\
    \midrule
    '''
    for period in ['December','January']:
        cost_table+=f'\\multicolumn{{6}}{{l}}{{\\textit{{{period}}}}} \\\\\n'
        for model in models:
            values=[]
            for weight in [.5,1,2,3,5]:
                r=cost[(cost.period==period)&(cost.cohort=='Low-history')&(cost.model==model)&(cost.weight==weight)].iloc[0]
                v=f'{r.loss:.4f}'
                values.append(r'\textbf{'+v+'}' if r['rank']==1 else v)
            cost_table+=model+' & '+' & '.join(values)+r' \\'+'\n'
        cost_table+=r'\midrule'+'\n' if period=='December' else ''
    cost_table+=r'''\bottomrule
    \end{tabular}
    \end{table}'''

    contribution=r'''The empirical contribution is a set of conditional comparisons for POI visit forecasting: adjacency performance given history available at the forecast origin, model rankings under asymmetric forecast preferences, and their subsequent-month replication. Reference-centered attributions describe relative feature-dependence profiles, while predictive value is assessed through forecast errors and controlled comparisons.'''

    blocks=[('Section 5.2.1：五种子与 matched comparison（放在多模型精度比较结尾）',matched),
            ('Section 5.3：预测前历史分组（现有 graph-transfer 结果之后）',transfer),
            ('Section 5.2.2：代价偏好分析（low-history 结果之后）',cost_method+'\n\n'+cost_table),
            ('Discussion：贡献与归因解释（一个简短段落）',contribution)]

    def mdtable(headers,rows):
        return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

    md='''# Forecast-time transfer 与非对称损失：新增结果及 LaTeX

    计算完成：2026-11-16（西雅图时间）。数值计算全部在 AutoDL4 CPU 完成，主统计约 41 秒；使用 64 个已保存预测文件，新增训练次数为 0。文中模型名统一采用 EvolveGCN。

    ## 1. 本次实验结论

    - **预测前分组**：训练期中位数固定分组，December 与 January 使用相同阈值。两套邻居定义、四个历史组、两个非 Identity 配置、两个月，共 32 项比较；Identity 平均 MAE 全部最低，31 项点态 95% 配对区间完全支持该方向，1 项包含零。这支持在本次 cohort、时期和 MAE 目标下采用 Identity；尚未形成需要另外部署评估的自动邻接选择规则。
    - **恢复与错误激活**：NAICS-4 的 low-target/high-neighbor 组增加了正访问事件上的预测总量，同时提高 FAR 和 MAE。恢复量与整体精度需要联合判断。
    - **代价偏好**：稀疏 POI 中，MTGNN 在低权重下更优；当低估权重为 5 时，EvolveGCN Identity 相对 MTGNN 的优势在两个月都得到配对区间支持。全部八种配置中，December 此时最低平均损失属于 Shared LSTM，January 属于 Identity。
    - **已有结果复现**：附录 A.1 八种配置的两个时期 MAE 均值与标准差全部复现。832 项独立数值检查通过，最大差异为 $8.9\\times10^{-16}$。

    ## 2. 实验口径

    | 项目 | 本次采用值 |
    |---|---|
    | 窗口 | $T=14,H=7$，完整训练及验证标签窗口 |
    | 训练 / 验证 | Oct. 1–Nov. 19 / Nov. 20–Dec. 3, 2020 |
    | 测试起点 | December 22 个，January 25 个；每个起点保留全部 7 个 leads |
    | 全体测试事件 | December 1,522,136；January 1,729,700 |
    | 随机种子 | 42–46，使用已选超参数；January 沿用冻结模型 |
    | 历史分组 cohort | 9,866 个 POI，在 NAICS-2 与 NAICS-4 下均有非自身邻居 |
    | 历史活动阈值 | Target 1.3571428571；NAICS-2 neighbor 2.9190666286；NAICS-4 neighbor 2.9496919918 visits/day |
    | 阈值来源 | 30 个完整训练起点及上述 POI 的历史均值；等于阈值归 Low |
    | Low-history cohort | 1,944 个 POI：前 50 天正训练均值不超过正均值分布的第 25 百分位（0.52），观测零值比例至少 70% |
    | 正事件 / 激活 | $Y>0$ / $\\widehat Y\\geq0.5$ |
    | 配对区间 | 固定五种子损失均值；2,000 次 7-origin 时间区块和 3 km 空间组联合重采样；点态 95% |

    同一组内三个 adjacency 使用相同事件。NAICS-2 与 NAICS-4 两种邻居定义分别产生一套分组。每个预测起点的分组只读取此前 14 天历史；January 使用当时已观测到的历史。重叠起点及 POI-day 事件的相关性由配对区块重采样处理。

    ## 3. 可直接加入论文的 LaTeX

    下列内容按现有章节插入。图文件 `forecast_time_transfer.pdf` 与论文源文件放在同一目录即可；使用论文已有的 `graphicx`、`amsmath`、`booktabs`。

    '''
    for title,body in blocks:
        md+='### '+title+'\n\n```latex\n'+body+'\n```\n\n'
    md+='## 4. 完整数值核对表\n\n以下是本次新增分析的完整聚合值，供核对和选择补充材料；正文建议使用上面的精简段落、图与表。\n\n'
    md+='### 4.1 按预测前历史分组的三配置比较\n\nRecall、FAR、DMR 均以百分比显示。DMR 为正访问事件上的预测总量 / 观测总量。Low/High 顺序为目标 / 邻居历史活动。\n\n'
    for period in ['December','January']:
        for graph in ['NAICS-2','NAICS-4']:
            md+=f'**{period}：{graph} 邻居定义**\n\n'
            rows=[]
            for cell in cells:
                for model in models[-3:]:
                    r=strata[(strata.period==period)&(strata.neighbor_partition==graph)&(strata.cell==cell)&(strata.model==model)].iloc[0]
                    rows.append([cell,model.replace('EvolveGCN ',''),f'{int(r.events):,}',f'{r.MAE:.4f}',f'{100*r.recall:.2f}',f'{100*r.FAR:.2f}',f'{100*r.DMR:.2f}'])
            md+=mdtable(['Target/Neighbor','Adjacency','Events','MAE','Recall (%)','FAR (%)','DMR (%)'],rows)+'\n'
    md+='### 4.2 非对称损失：全部模型、两种 cohort\n\n粗体为该时期、cohort 和权重下最低平均损失。\n\n'
    for cohort in ['All','Low-history']:
        for period in ['December','January']:
            md+=f'**{period} / {cohort}**\n\n';rows=[]
            for model in models:
                vals=[]
                for weight in [.5,1,2,3,5]:
                    r=cost[(cost.period==period)&(cost.cohort==cohort)&(cost.model==model)&(cost.weight==weight)].iloc[0]
                    value=f'{r.loss:.5f}';vals.append('**'+value+'**' if r['rank']==1 else value)
                rows.append([model]+vals)
            md+=mdtable(['Model','0.5','1','2','3','5'],rows)+'\n'
    md+='### 4.3 Low-history 的恢复与错误激活\n\n'
    for period in ['December','January']:
        md+='**'+period+'**\n\n';rows=[]
        for model in models:
            r=recovery[(recovery.period==period)&(recovery.cohort=='Low-history')&(recovery.model==model)].iloc[0]
            rows.append([model,f'{100*r.recall:.2f}',f'{100*r.FAR:.2f}',f'{100*r.DMR:.2f}'])
        md+=mdtable(['Model','Recall (%)','FAR (%)','DMR (%)'],rows)+'\n'
    md+='### 4.4 稀疏 cohort 中 MTGNN − Identity 的配对损失差\n\n正值表示 EvolveGCN Identity 损失较低；负值表示 MTGNN 损失较低。\n\n'
    rows=[]
    for _,r in pair[pair.cohort=='Low-history'].iterrows():rows.append([r.period,f'{r.weight:g}',f'{r.estimate:+.5f}',f'[{r.ci_low:+.5f}, {r.ci_high:+.5f}]'])
    md+=mdtable(['Month','Weight','Difference','Pointwise 95% interval'],rows)
    md+='''
    ## 5. 可复核文件

    - [训练期阈值](evidence/thresholds.json)
    - [四组全部均值与五种子标准差](evidence/strata_summary.csv)
    - [全部 graph–Identity 配对区间](evidence/strata_gain_intervals.csv)
    - [各权重损失、标准差、低估与高估分量、排名](evidence/cost_summary.csv)
    - [恢复、错误激活和幅度恢复](evidence/recovery_summary.csv)
    - [损失差配对区间](evidence/cost_pair_intervals.csv)
    - [冻结协议](evidence/PROTOCOL.md)、[计算脚本](evidence/analyze.py)、[独立复核脚本](evidence/verify.py)
    - [源文件哈希与审计](evidence/audit.json)、[独立复核结果](evidence/verification.json)
    - [论文图 PDF](forecast_time_transfer.pdf)、[图预览](forecast_time_transfer.png)

    AutoDL4 计算与输出目录：`/root/poi_decision_analysis_20261116/`。保存的结果包含每个种子的明细；所有源预测、检查点与数据文件保持原哈希。该目录新增文件体积小于 1 MB。当前交付为新增 MD、图和核验依据。
    '''
    (OUT/'FORECAST_TIME_AND_COST_RESULTS.md').write_text(md,encoding='utf-8')
    validation=OUT/'latex_validation';validation.mkdir(exist_ok=True)
    shutil.copy2(OUT/'forecast_time_transfer.pdf',validation/'forecast_time_transfer.pdf')
    tex=r'''\documentclass[11pt]{article}
    \usepackage[margin=1in]{geometry}
    \usepackage{amsmath,booktabs,graphicx,caption}
    \usepackage[T1]{fontenc}
    \begin{document}
    '''
    for title,body in blocks:
        tex+='\n'+body+'\n\clearpage\n'
    tex+=r'\end{document}'
    (validation/'additions.tex').write_text(tex,encoding='utf8')
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(str(OUT))


if __name__ == '__main__':
    main()
