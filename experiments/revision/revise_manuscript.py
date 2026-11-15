"""Apply evidence-backed corrections to the preserved Overleaf source."""
from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'paper/original/outline.tex'
DEST = ROOT/'paper/revision/outline.tex'
original = SOURCE.read_text(encoding='utf8')
text = original.replace('EvolveGCN', 'GraphGRU')


def replace(old, new):
    global text
    assert text.count(old) == 1, (old[:100], text.count(old))
    text = text.replace(old, new)


def between(start, end, body):
    global text
    assert text.count(start) == text.count(end) == 1
    a, b = text.index(start), text.index(end)
    text = text[:a] + body.rstrip() + '\n\n' + text[b:]


replace('Evolving Graph Convolutional Networks (GraphGRU) and Multivariate Time Series Graph Neural Networks (MTGNN)',
        'a graph-augmented gated recurrent unit (GraphGRU) and a Multivariate Time Series Graph Neural Network (MTGNN) with fixed adjacency')
between('Representative architectures include', 'For urban planners and transportation agencies,',
r'''Representative architectures include Evolving Graph Convolutional Networks (EvolveGCN), which update graph-convolution parameters across input days \citep{pareja2020evolvegcn}, and Multivariate Time Series Graph Neural Networks (MTGNN), which combine graph learning with temporal convolutions \citep{wu2020connecting}.
We compare a graph-augmented gated recurrent unit (GraphGRU) with a fixed-adjacency MTGNN to examine how temporal modeling and supplied POI relationships affect forecasts.''')
replace('For example, visits to a clinic may be driven more by its functional identity than by its location, so opening a restaurant at the same site may not necessarily attract the same level of customer traffic.',
        'For example, a clinic forecast may depend more on its recorded functional identity than on its coordinates.')
replace('Therefore, understanding what drives predicted visits can help decision makers assess whether demand is tied to the location itself or to the specific type of POI operating there.',
        'Attribution can show which recorded inputs support a forecast; estimating the effect of changing a POI requires a separate behavioral analysis.')
replace(r'The dataset contains daily observed visits from October to December 2020, covering \(92\) days and \(N=9{,}884\) POIs.',
        r'The dataset contains daily observed visits from October to December 2020, covering \(92\) days and \(N=9{,}884\) POIs. A zero denotes no visits recorded in the panel for a POI-day and does not establish that no visits occurred outside the panel.')
replace(r'The LSTM baseline \citep{hochreiter1997long} captures nonlinear temporal dependencies in historical visit sequences.',
        r'The LSTM baseline \citep{hochreiter1997long} processes historical visit sequences recurrently.')
between(r'\textit{(1) Historical Average.}', r'\textit{(2) AR-ridge.}',
r'''\textit{(1) Historical Average.}
The historical-average (HA) predictor uses the observed visits in the training input windows.
Let \(\mathcal T_{\mathrm{tr}}\) denote the training calendar, \(\mathcal T_{\mathrm{tr},j}\) its dates with an observed count for POI \(j\), and \(y_{\tau,j}\) that count.
If \(\omega_\tau\) is the number of training input windows containing day \(\tau\), the fitted-run reference is
\(\bar y_j^{\mathrm{hist}}=\frac{\sum_{\tau\in\mathcal T_{\mathrm{tr},j}}\omega_\tau y_{\tau,j}}{\sum_{\tau\in\mathcal T_{\mathrm{tr},j}}\omega_\tau}\).
Overlapping windows therefore weight calendar days differently.
We also evaluate an equally weighted calendar mean, a same-weekday training mean, the last observation, a repeated last-week forecast, and a recent mean whose window is selected by validation MAE from 7, 14, and 28 days subject to the available input length.
These additional benchmarks use the same scored test events as the fitted models.
We also fit NLinear \citep{zeng2023transformers} with weights shared across POIs. It subtracts the last observation before the affine forecast and restores that observation afterward; we solve its ridge-regularized squared-error objective directly and select the penalty from \(\{0,10^{-4},10^{-2},1\}\) by validation MAE.
The Teunter--Syntetos--Babai (TSB) benchmark \citep{teunter2011intermittent} separately smooths positive-visit size and visit-occurrence probability, selecting both smoothing constants from \(\{0.05,0.1,0.2\}\) on validation MAE.''')
between(r'\subsubsection{GraphGRU}', r'\subsubsection{MTGNN}',
r'''\subsubsection{GraphGRU}
\label{subsubsec:evolvegcn}
GraphGRU applies a fixed graph operator to historical visits, projects each POI's inputs to 32 dimensions, and processes the sequence with a shared gated recurrent unit (GRU).
For functional graphs, the propagated input is the average of the target count and its group mean, including the target in that mean.
The decoder combines the final recurrent state with a linear projection of the target's own last observation to produce \(H\) forecasts.
The graph operator is fixed; the recurrent state evolves across input days.

The GNN-static specification uses this same recurrence with identity adjacency and adds static attributes to the input projection at each time step.
Its 52 static dimensions comprise four numeric inputs and six learned eight-dimensional embeddings for category, NAICS-2, NAICS-4, brand, parent place, and sub-category.
The numeric inputs are latitude, longitude, opening hours, and an opening-hour missingness indicator.
Brand and parent-place labels appearing at most 20 times are pooled; missing labels form a separate category. Other categorical labels appearing only once are pooled.''')
between('We use the multistep forecasting implementation of MTGNN', r'\subsection{Adjacency Matrices}',
r'''We use the multistep implementation with graph learning disabled and the supplied adjacency held fixed.
Both graph-propagation directions use the selected relation, while temporal convolutions model lagged dependence.
For functional graphs, propagation averages the current node representation with its group mean; identity adjacency leaves it unchanged.
The implementation normalizes across channels and time within each node, so identity adjacency does not mix POI histories through normalization.
The comparison therefore evaluates two temporal architectures under specified relationships.''')
between('To capture changes in visit-pattern similarity over time,', r'\subsection{Forecast Evaluation}',
r'''All reported visit-correlation graphs use training-period observations and remain fixed during validation and testing.''')
replace('summed squared deviations from the observed mean at each forecast horizon.',
        'summed squared deviations from the pooled test mean.')
replace('All metrics are computed across test samples and POIs for each forecast horizon.',
        'Metrics pool scored events across forecast origins, leads, and POIs within each setting.')
replace(r'\frac{1}{|\mathcal{T}_{\mathrm{tr}}|}'+'\n'+r'\sum_{\tau\in\mathcal{T}_{\mathrm{tr}}}',
        r'\frac{1}{|\mathcal{T}_{\mathrm{tr},j}|}'+'\n'+r'\sum_{\tau\in\mathcal{T}_{\mathrm{tr},j}}')
replace('The \\textit{persistence-density} approach combines the overall nonzero ratio with the probability of consecutive positive-visit days,',
        'The \\textit{persistence-density} approach ranks POIs by the product of the overall nonzero ratio and the probability of consecutive positive-visit days,')
replace(r'where \(\mathcal{T}_{\mathrm{tr}}^{-}\) contains all training days except the last day.',
        r'where \(\mathcal{T}_{\mathrm{tr}}^{-}\) contains training days with both the current and following count observed; the probability is zero when its denominator is zero.')
replace(r'where \(z_j=1-r_j\) denotes the training-period zero-visit ratio, and \(Q_{\alpha}\) is the \(\alpha\)-quantile of positive historical averages across POIs.',
        r'where \(z_j=\frac{\sum_{\tau\in\mathcal T_{\mathrm{tr},j}}\omega_\tau\mathbb I\{y_{\tau,j}=0\}}{\sum_{\tau\in\mathcal T_{\mathrm{tr},j}}\omega_\tau}\) is the zero-visit ratio over observed training-window inputs, and \(Q_{\alpha}\) is the \(\alpha\)-quantile of positive historical averages across POIs. '
        r'The resulting cohort depends on \(T\); an equally weighted calendar definition identifies 2,004 POIs and is reported separately in the reproducibility outputs.')
replace('Lift measures the average increase in predicted visits relative to the historical-average (HA) baseline,',
        'Lift measures the average increase in predicted visits relative to the HA baseline,')
replace('We use three diagnostic metrics to characterize model predictions for these events.',
        'We use three diagnostic metrics to characterize model predictions for these events. DMR describes forecast magnitude on positive events; event detection also depends on how forecasts separate positive and zero visits.')
replace('measures the change in predicted visits associated with cross-POI information transfer, which we refer to as \\textit{graph lift}.',
        'is termed \\textit{graph lift}. The two models are fitted separately, so this contrast includes changes in learned parameters as well as adjacency. The subsequent fixed-model perturbations isolate sensitivity to neighbor inputs.')
replace('Table~\\ref{tab:graph_transfer_regime_definitions} summarizes the resulting graph-transfer regimes.',
        'Table~\\ref{tab:graph_transfer_regime_definitions} summarizes the resulting graph-transfer regimes. Events are assigned in the order spurious, substitutive, confirmatory, redundant, and conservative zero; unassigned events form Other. Because realized visits enter these definitions, the regimes describe forecast outcomes after observation.')
replace('Using MTGNN with identity adjacency, we compare the attribution of four semantic descriptors:',
        'Using GNN-static with identity adjacency, we compare the attribution of four semantic descriptors:')
replace('The score combines the input value with the local sensitivity of the prediction to that input.',
        r'The score combines the encoded input with local prediction sensitivity. Numeric static inputs retain their original units in these fits. Translating a coordinate by \(c\) while compensating the first-layer bias preserves predictions but changes its attribution by \(c\,\partial\hat Y/\partial x\); location shares therefore depend on the coordinate origin. Categorical scores are computed on learned embedding components.')
replace('Table~\\ref{tab:parameter_settings} summarizes the main experimental settings.',
r'''The training and validation losses exclude labels beyond their respective period ends, including when an output window straddles a boundary.
Normalization uses observed training input windows; missing lag values enter as zero placeholders and missing labels are unscored.
The original neural fits use seed 42, early stopping with patience 10, and validation-selected checkpoints.
GraphGRU and LSTM minimize squared error; MTGNN minimizes absolute error, so their recovery contrast also reflects the loss function.
For \(H=14\) and \(H=28\), only the first seven leads enter validation; these are exploratory extrapolation results.
The 7-day setting supplies the primary short-horizon comparison.
Table~\ref{tab:revision_protocol} reports the origin counts, and Table~\ref{tab:parameter_settings} summarizes the model settings.
\input{revision_protocol.tex}''')
replace('The top-category and NAICS-4 adjacencies produce identical results in this comparison.',
        'Top category and NAICS-4 induce exactly the same partition of the 9,884 POIs into 134 groups, so their identical results represent one graph specification. NAICS-2 forms 22 groups; its mean non-self degree is 1,586.5, compared with 767.0 for NAICS-4. The comparison therefore changes both semantic granularity and neighborhood size.')
replace('In the spurious regime, a larger mean graph lift of \\(0.77\\) visits corresponds to an MAE gain of \\(-0.77\\) visits, indicating that information from active neighbors can substantially overestimate demand at locations with no realized visits.',
        'In the spurious regime, mean graph lift is \\(0.77\\) visits and MAE gain is \\(-0.77\\) visits. Their opposite signs follow algebraically from the zero outcome and nonnegative forecasts used to define this regime.')
replace('This suggests that parent-place information provides a distinctive predictive signal for a relatively small subset of POIs, despite its limited overall contribution.',
        'This identifies unusually high parent attribution relative to the cohort distribution, while its original share remains small.')
replace('The analysis covers 581 POIs with recorded parent places and compares attribution patterns across different embedded POI groups.',
        'The analysis covers 581 POIs with 233 recorded parent places. Frequency pooling retains two parent labels separately and combines the remaining 231, so the attribution primarily describes this coarse parent-place encoding. We compare its attribution across embedded POI groups.')
replace('This pattern suggests that parent-place context serves as a supplementary signal, while individual visit histories and other attributes remain central to interpreting establishment-level demand forecasts.',
        'This pattern describes the supplementary contribution of the pooled parent-place encoding; it does not distinguish the effects of individual low-frequency parent places.')
replace('This concentration indicates that broad industry classifications provide the most prominent semantic signal for most eligible predictions, despite the simultaneous inclusion of more detailed descriptors.',
        'This concentration describes attribution within the jointly encoded representation; the overlapping semantic descriptors do not provide independent estimates of predictive value.')
replace('This perspective is important because sparse POIs may be systematically underestimated by conservative baselines, while graph-based models may improve demand recovery only when contextual information is transferred in a reliable manner.',
        'Sparse POIs expose a tradeoff between recovering intermittent positive visits and avoiding false activations.')
replace('The framework evaluates two representative temporal graph models, MTGNN and GraphGRU,',
        'The framework evaluates fixed-adjacency MTGNN and GraphGRU')
# The replacement above includes its following space in the original sentence.
replace('predictive signal, while broad industry classifications dominate semantic attribution',
        'attribution component, while broad industry classifications dominate semantic attribution')
replace('These findings suggest that explainable POI forecasting should evaluate not only prediction error, but also the reliability and source of contextual information used by the model.',
        'These diagnostics identify model sensitivities in the Seattle 2020 sample. Their stability across seeds, feature encodings, cities, and periods requires further evaluation.')

data = pd.read_csv(ROOT/'artifacts/revision_20261114/baseline_metrics.csv')
extended = pd.read_csv(ROOT/'artifacts/revision_20261114/extended_baseline_metrics.csv')
neural = pd.read_csv(ROOT/'artifacts/archived_2020q4/exp1/multi_horizon_results.csv')
settings = [(14,7),(28,7),(28,14),(28,28)]
table = [r'\begin{table}[!htbp]',r'\centering\small',
         r'\caption{Additional benchmarks on the same scored test events. Configurations are selected by validation MAE. NLinear uses a shared affine map fitted by ridge regression. Calendar HA gives each observed training date equal weight.}',
         r'\label{tab:revision_baselines}',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
         r'\((T,H)\) & Calendar HA & Weekday HA & Last week & Recent mean & NLinear & TSB \\',r'\midrule']
gains = []
for t,h in settings:
    rows = data[(data['T']==t)&(data.H==h)&(data.cohort=='All')].set_index('model')
    new = extended[(extended['T']==t)&(extended.H==h)&(extended.cohort=='All')].set_index('model')
    values = [rows.loc[m,'MAE'] for m in ['Calendar HA','Weekday HA','Seasonal naive','Recent mean']]
    values += [new.loc[m,'MAE'] for m in ['NLinear ridge','TSB']]
    table.append(f'({t},{h}) & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    mt = neural[(neural['T']==t)&(neural.H==h)&(neural.model=='MTGNN')].mae.iloc[0]
    gains.append(100*(1-mt/new.loc['NLinear ridge','MAE']))
table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(DEST.parent/'revision_baselines.tex').write_text('\n'.join(table)+'\n',encoding='utf8')
addition = (r'\input{revision_baselines.tex}'+'\n\n'+
    r'Table~\ref{tab:revision_baselines} provides stronger simple references. '
    r'The validation-selected recent mean has MAE 1.343 at \(H=7\), below the identity GraphGRU values of 1.37--1.38. '
    rf'NLinear further reduces MAE to 1.264 and 1.231 in the two 7-day settings. MTGNN remains more accurate than NLinear, with reductions of {min(gains):.1f}\%--{max(gains):.1f}\% across the four settings. '
    r'The stronger temporal baselines narrow the advantage over HA; aggregate accuracy gains beyond these benchmarks are concentrated in MTGNN.'+'\n\n')
anchor = '\\subsubsection{Low-History POIs}'
text = text.replace(anchor,addition+anchor)
protocol = [r'\begin{table}[!htbp]',r'\centering\small',
            r'\caption{Forecast-window audit. Training and validation counts include boundary windows with masked labels. Scored target dates are disjoint across splits.}',
            r'\label{tab:revision_protocol}',r'\begin{tabular}{lrrrl}',r'\toprule',
            r'\((T,H)\) & Train origins & Validation origins & Test origins & Validated leads \\',r'\midrule']
for t,h in settings:
    protocol.append(f'({t},{h}) & {57-t} & 7 & {29-h} & 1--7'+r' \\')
protocol += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(DEST.parent/'revision_protocol.tex').write_text('\n'.join(protocol)+'\n',encoding='utf8')
for filename in ('recovery_activation_tradeoff.pdf','forecasting_performance_across_horizons.pdf'):
    text = text.replace('art/'+filename,filename)
assert re.findall(r'\\section\{[^}]+\}',text)==re.findall(r'\\section\{[^}]+\}',original)
assert '\\begin{abstract}' in text and '\n\n' not in text.split('\\begin{abstract}')[1].split('\\end{abstract}')[0].strip()
DEST.write_text(text,encoding='utf8')
bib=(SOURCE.parent/'reference.bib').read_text(encoding='utf8')
entry='''\n@inproceedings{zeng2023transformers,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37}, pages={11121--11128}, year={2023},
  doi={10.1609/aaai.v37i9.26317}
}\n'''
if 'zeng2023transformers' not in bib:
    bib += entry
(DEST.parent/'reference.bib').write_text(bib,encoding='utf8')
print(json.dumps({'section_titles_preserved':True,'NLinear_comparison_percent':gains,'output':str(DEST)}))
