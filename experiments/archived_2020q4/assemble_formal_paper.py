"""Build the current manuscript front and compact experiment tables."""
from argparse import ArgumentParser
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd


ATTRIBUTION_SOURCES = ("History", "Location", "Identity", "Parent", "Others")
ATTRIBUTION_REGIMES = tuple(f"{source}-driven" for source in ATTRIBUTION_SOURCES)
REGIME_CAPTIONS = {
    "fig:attribution_fingerprint": (
        r"Mean standardized attribution shares by regime for the full-feature GNN-static fit at \(T=14\), \(H=7\). "
        r"The color scale is centered at zero. Bold values mark the feature group "
        r"that defines each regime. Bars show the number of POIs."
    ),
    "fig:attribution_regime_map": (
        r"Spatial distribution of the attribution regimes in Figure~\ref{fig:attribution_fingerprint} "
        r"among @MAP_N@ POIs within the map extent. The overview shows all regimes. Each smaller panel "
        r"highlights one regime in color and shows the remaining POIs in gray. "
        r"Roads and water polygons are from the 2020 U.S. Census Bureau TIGER/Line files."
    ),
}

FIGURES = [
    ("exp0", "poi_sparsity_distribution.pdf", "fig:poi_sparsity_distribution", "After fig:visit_overview (original line 391).",
     r"Visit sparsity among 9,884 POIs from October 1 to December 31, 2020. Panel (a) shows daily counts for 905,918 observed POI--days. Panel (b) shows the fraction of observed days with positive visits for each POI. The dashed line marks the median."),
    ("exp0", "exp_logic.pdf", "fig:exp_logic", "Move from original lines 784--789 to before Forecasting Models (line 474).",
     r"Two graph neural network (GNN) formulations for POI demand forecasting. GNN-adj varies adjacency and perturbs neighbor histories. GNN-static uses gradient attribution to group POIs by the information used in their forecasts."),
    ("exp1", "forecasting_performance_across_horizons.pdf", "fig:forecast_horizon_skill", "After tab:multihorizon (original lines 886--887).",
     r"Mean absolute error (MAE) skill relative to the historical average at \(T=28\) days. The best MTGNN and non-identity EvolveGCN configurations are selected by validation loss within each horizon. The 28-day horizon has one test forecast origin."),
    ("exp2", "recovery_activation_tradeoff.pdf", "fig:recovery_activation_tradeoff", "After tab:zero_collapse_diagnostic (original line 1002).",
     r"Demand mass recovery ratio (DMR) and false activation rate (FAR) among @REC_N@ low-history POIs at \(T=14\), \(H=7\). The activation threshold is 0.5 visits. Labels identify adjacency matrices and benchmarks."),
    ("exp3", "outcome_conditioned_graph_transfer.pdf", "fig:counterfactual_graph_transfer_map", "After the updated tab:graph_transfer_summary (original line 1103). Replace the old PNG block at lines 1124--1129.",
     r"MAE gain of NAICS-2 over identity adjacency at \(T=28\), \(H=14\). Positive gains indicate lower error under NAICS-2. Both panels use pooled evidence sextiles and a common zero-centered color scale. Axis ticks mark bin edges. Panel means weight cells by event count. Cells with fewer than 30 events are marked with a dash."),
    ("exp4", "neighbor_perturbation_tradeoff.pdf", "fig:neighbor_perturbation_tradeoff", "After tab:metric_targeted_perturbation_naics4 (original line 1260).",
     r"Effect of zeroing neighbor histories under NAICS-4 adjacency at \(T=28\), \(H=14\). Each pair gives the share of false activations removed and the share of positive-demand recoveries lost in their respective reference sets."),
    ("exp5", "attribution_fingerprint.pdf", "fig:attribution_fingerprint", "Before tab:attri_regime (original lines 1318--1324).",
     r"Mean attribution shares for five clusters from the full-feature GNN-static fit at \(T=14\), \(H=7\). Cells show the original percentages on a square-root color scale. Bold values mark the largest share in each row. Bars show cluster sizes."),
    ("exp5", "attribution_cluster_map.pdf", "fig:attribution_regime_map", "Replace the old map at original lines 1452--1457, after the category cluster table.",
     r"Spatial concentration of the five attribution clusters in Figure~\ref{fig:attribution_fingerprint} among @MAP_N@ POIs within the map extent. Outlined cells contain at least 60 POIs on a 2 km grid. Color marks the cluster with the strongest evidence of overrepresentation in cells with Holm-adjusted \(p<0.05\). The adjustment covers one-sided hypergeometric tests for all outlined cells and clusters. Cell labels give the excess above the mapped share in percentage points (pp). The curves give the mean same-cluster share among the indicated number of nearest other POIs minus its expectation under random labels with fixed cluster sizes. Water polygons are from the 2020 U.S. Census Bureau TIGER/Line files."),
]


def escape(value):
    return re.sub(r"[&%$#_{}\\~^]", lambda match: {
        "\\": r"\textbackslash{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }.get(match[0], "\\"+match[0]), str(value)).replace("'", r"\textquotesingle{}")


def number(value, places=2):
    return "--" if pd.isna(value) else f"{value:.{places}f}"


def percent(value, places=1):
    return number(100*value, places)+r"\%"


def table(label, caption, headings, rows, column_format):
    content = "\n".join(" & ".join(str(value) for value in row)+r" \\" for row in rows)
    return "\n".join([
        r"\begin{table}[!htbp]", r"\centering", r"\small", r"\caption{"+caption+"}",
        r"\label{"+label+"}", r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{"+column_format+"}", r"\toprule",
        " & ".join(headings)+r" \\", r"\midrule", content, r"\bottomrule",
        r"\end{tabular}}", r"\end{table}",
    ])


def figure(name, label, caption, path="art"):
    return "\n".join([
        r"\begin{figure}[!htbp]", r"\centering", r"\includegraphics[width=\linewidth]{"+path+"/"+name+"}",
        r"\caption{"+caption+"}", r"\label{"+label+"}", r"\end{figure}",
    ])


def formatted_tables(data):
    def load(name):
        return pd.read_csv(data/name, keep_default_na=False, na_values=[""])
    mh = load("multi_horizon_results.csv")
    rec = load("01_recovery_activation.csv")
    graph = load("02_graph_transfer_regimes.csv")
    pert = load("03_neighbor_perturbation.csv")
    clusters = load("04_attribution_clusters.csv")
    cats = load("category_cluster_distribution.csv")
    parent = load("parent_attribution.csv")
    sem = load("semantic_attribution.csv")
    dens = load("density_sensitivity.csv")
    tables = {}
    tables["tab:multihorizon"] = table(
        "tab:multihorizon",
        r"Multi-horizon forecasting diagnostics. For each \((T,H)\), the best MTGNN and non-identity EvolveGCN adjacencies are selected by their validation losses. The skill column reports \(100(1-\mathrm{MAE}_{\mathrm{model}}/\mathrm{MAE}_{\mathrm{HA}})\). Anomaly correlation uses deviations from the historical average.",
        [r"\((T,H)\)", "Model", "Adjacency", "MAE", "RSE", "CORR", r"Skill (\%)", "Anom. CORR",
         r"\(\Delta\mathrm{DMR}\)", r"\(\mathrm{FAR}_{0.5}\)"],
        [(f"({r['T']},{r['H']})", r.model, "--" if r.adjacency=="None" else r.adjacency,
          *[number(r[k]) for k in ("mae", "rse", "corr", "skill_percent", "anomaly_corr", "delta_dmr", "far_0_5")])
         for _, r in mh.iterrows()],
        "lllrrrrrrr",
    )
    tables["tab:sublevel"] = table(
        "tab:sublevel", r"Visit-density sensitivity at \(T=28\), \(H=14\). Skill is relative to the historical average within each subset.",
        ["Subset", "POIs", "Nonzero ratio", "EvolveGCN MAE", "HA MAE", r"Skill (\%)", "RSE", "CORR"],
        [(r.subset, f"{r.n_pois:,}", *[number(getattr(r,k)) for k in
          ("nonzero_ratio", "mae", "ha_mae", "skill_percent", "rse", "corr")]) for r in dens.itertuples()],
        "lrrrrrrr",
    )
    tables["tab:zero_collapse_diagnostic"] = table(
        "tab:zero_collapse_diagnostic",
        r"Zero-demand collapse at \(T=14\), \(H=7\). MAE and root mean squared error (RMSE) use all test POIs. Other columns use low-history POIs.",
        ["Model", "Adjacency", "MAE", "RMSE", "Lift", r"\(\mathrm{DMR}^{\mathrm{model}}\)",
         r"\(\mathrm{DMR}^{\mathrm{HA}}\)", r"\(\Delta\mathrm{DMR}\)", r"\(\mathrm{FAR}_{0.5}\)", "Mean pred."],
        [(r.model, "--" if r.adjacency=="None" else r.adjacency, *[number(getattr(r,k)) for k in
          ("mae","rmse","lift","dmr_model","dmr_ha","delta_dmr","far_0_5","mean_prediction")])
         for r in rec.itertuples()], "llrrrrrrrr",
    )
    graph_rows = ("graph_lift", "target_evidence", "neighbor_evidence", "mean_mae_gain")
    tables["tab:graph_transfer_summary"] = table(
        "tab:graph_transfer_summary",
        r"Graph-transfer regimes under NAICS-2 at \(T=28\), \(H=14\). Positive and false activation shares use all events in each regime.",
        ["Regime", "Events", "Graph lift", "Target evidence", "Neighbor evidence", "MAE gain", "Positive / false share"],
        [(r.regime, f"{r.n_events:,}", *[number(getattr(r,k)) for k in graph_rows],
          number(r.positive_event_share)+" / "+number(r.false_activation_share))
         for r in graph.loc[graph.adjacency.eq("NAICS-2")].itertuples()], "lrrrrrr",
    )
    tables["tab:semantic_granularity_transfer"] = table(
        "tab:semantic_granularity_transfer",
        r"Substitutive and spurious transfer under functional adjacencies at \(T=28\), \(H=14\).",
        ["Adjacency", "Regime", "Events", "Graph lift", "MAE gain", "Positive / false share"],
        [(r.adjacency, r.regime.removesuffix(" transfer"), f"{r.n_events:,}", number(r.graph_lift),
          number(r.mean_mae_gain), number(r.positive_event_share)+" / "+number(r.false_activation_share))
         for r in graph.loc[graph.regime.isin(["Substitutive transfer","Spurious transfer"])].itertuples()], "llrrrr",
    )
    tables["tab:metric_targeted_perturbation_naics4"] = table(
        "tab:metric_targeted_perturbation_naics4",
        r"Neighbor-group perturbation under NAICS-4 at \(T=28\), \(H=14\). Ben. and harm. refer to beneficial and harmful attempted-substitution events. Positive MAE effects indicate that zeroing the group increases error.",
        ["Neighbor group", "Lift (ben.)", "Lift (harm.)", "Per node (ben.)", "Per node (harm.)",
         r"\(\Delta\)MAE (ben.)", r"\(\Delta\)MAE (harm.)", "FA removed", "Rec. lost"],
        [(r.neighbor_group, *[number(getattr(r,k),3) for k in
          ("beneficial_lift","harmful_lift","beneficial_per_node","harmful_per_node","beneficial_mae_effect",
           "harmful_mae_effect","false_activations_removed","recoveries_lost")]) for r in pert.itertuples()], "lrrrrrrrr",
    )
    labels = clusters.display_label.tolist()
    named = tuple(labels) == ATTRIBUTION_REGIMES
    representatives = [
        "--" if pd.isna(value) or not value else
        r"\newline ".join(escape(name.strip()) for name in value.split(";"))
        for value in clusters.representative_pois
    ]
    if named:
        tables["tab:attri_regime"] = table(
            "tab:attri_regime",
            r"Attribution regimes from the GNN-static fit at \(T=14\), \(H=7\). "
            r"Representative POIs have attribution profiles closest to the mean profile of their regime.",
            ["Regime", "POIs", "Representative POIs", "MAE"],
            [(r.display_label, f"{r.n_pois:,}", names, number(r.mae,3))
             for r, names in zip(clusters.itertuples(), representatives)], "lrp{7.5cm}r",
        )
        category_caption = (
            r"Attribution-regime shares (\%) by broad POI category. Each source column gives the share "
            r"of POIs assigned to the corresponding regime in Table~\ref{tab:attri_regime}. "
            r"MAE uses the same GNN-static predictions."
        )
        category_headings = ATTRIBUTION_SOURCES
    else:
        tables["tab:attri_regime"] = table(
            "tab:attri_regime",
            r"Attribution clusters from the GNN-static fit at \(T=14\), \(H=7\). Representative POIs are nearest to the corresponding attribution centroid.",
            ["Cluster", "Dominant source", "POIs", "Representative POIs", "MAE"],
            [(r.display_label.split(":")[0], r.dominant_source, f"{r.n_pois:,}", names, number(r.mae,3))
             for r, names in zip(clusters.itertuples(), representatives)], "llrp{7.5cm}r",
        )
        category_caption = (
            r"Attribution-cluster shares by broad POI category. Columns 1--5 give the percentages assigned to the clusters in Table~\ref{tab:attri_regime}. MAE uses the same GNN-static predictions."
        )
        category_headings = [label.split(":")[0].removeprefix("Cluster ") for label in labels]
    tables["tab:category_regime_mae"] = table(
        "tab:category_regime_mae", category_caption,
        ["POI category", "POIs", *category_headings, "MAE"],
        [(escape(r["category9"]), f"{r['n_pois']:,}", *[number(100*r[k],1) for k in labels], number(r["mae"],3))
         for _, r in cats.iterrows()], "lrrrrrrr",
    )
    short_parent = {
        "Cafes and restaurants inside larger destinations": "Cafes and restaurants",
        "Pharmacies and health-related retail inside medical or commercial places": "Pharmacies and health retail",
        "Retail and service shops inside shopping centers or office buildings": "Retail and service shops",
        "Education-related POIs inside campuses": "Education-related POIs",
        "Medical and clinic-related POIs inside larger health facilities": "Medical and clinic-related POIs",
        "All embedded POIs": "All embedded POIs",
    }
    tables["tab_parent"] = table(
        "tab_parent", r"Attribution shares for POIs with a recorded parent place. Subgroups use the top category of the embedded POI. Gap is the parent share minus the combined share of the remaining sources.",
        ["Embedded POI group", "POIs", r"Parent (\%)", r"Own (\%)", "Gap (pp)", "MAE"],
        [(short_parent[r.group], f"{r.n_pois:,}", number(100*r.parent_share,1), number(100*r.own_share,1),
          number(100*r.gap,1), number(r.mae,3)) for r in parent.itertuples()], "lrrrrr",
    )
    tables["tab:semantic_granularity"] = table(
        "tab:semantic_granularity",
        r"Attribution of four semantic descriptors in the GNN-static fit. Dominance is assessed among the four descriptors for events with nonzero semantic attribution.",
        ["Descriptor", r"Mean share (\%)", r"Dominance (\%)", "Dominant-event MAE", "Dominant events"],
        [(r.descriptor, number(100*r.mean_share,2), number(100*r.dominance_rate,2),
          number(r.dominant_mae,3), f"{r.n_dominant:,}") for r in sem.itertuples()], "lrrrr",
    )
    return tables, (mh, rec, graph, pert, clusters, cats, parent, sem, dens)


def replace_between(text, start, end, new):
    left = text.index(start)
    right = text.index(end, left+len(start))
    return text[:left] + new + text[right:]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def build_front(source, data, art, panel_path):
    manuscript = source.read_text(encoding="utf-8-sig")
    boundary = r"\section{Feature attribution for neural networks}"
    assert manuscript.count(boundary) == 1 and manuscript[:manuscript.index(boundary)].count("\n") == 1616
    text = manuscript.split(boundary, 1)[0]
    tables, frames = formatted_tables(data)
    mh, rec, graph, pert, clusters, cats, parent, sem, dens = frames
    named = tuple(clusters.display_label) == ATTRIBUTION_REGIMES
    protocol = json.loads((data/"forecast_protocol.json").read_text(encoding="utf-8"))
    diag = protocol["T14_H7"]
    graph_stats = json.loads((data/"graph_transfer_summary.json").read_text(encoding="utf-8"))["NAICS-2"]
    perturb_stats = json.loads((data/"perturbation_summary.json").read_text(encoding="utf-8"))
    naics_equal = graph.loc[graph.adjacency.eq("NAICS-4")].drop(columns="adjacency").reset_index(drop=True).equals(
        graph.loc[graph.adjacency.eq("Top category")].drop(columns="adjacency").reset_index(drop=True))
    with ZipFile(panel_path) as panel:
        coords = pd.read_csv(BytesIO(panel.read("poi_attributes.csv")), usecols=["latitude","longitude"])
    plotted = int((coords.longitude.between(-122.470,-122.235) & coords.latitude.between(47.490,47.764)).sum())
    tokens = {"@REC_N@": f"{diag['low_history_pois']:,}", "@MAP_N@": f"{plotted:,}"}
    figures = {}
    for _, name, label, _, caption in FIGURES:
        assert (art/name).is_file()
        if named:
            caption = REGIME_CAPTIONS.get(label, caption)
        for key, value in tokens.items():
            caption = caption.replace(key, value)
        figures[label] = figure(name, label, caption)
    f = figures.__getitem__
    tab = tables.__getitem__
    start = r"\section{Empirical Results}"
    best_at_28 = mh.loc[mh["T"].eq(28)].sort_values("mae").groupby("H").first()
    mtgnn_wins = best_at_28.model.eq("MTGNN").all()
    headline = (
        r"We analyze 92 days of SafeGraph visit data for 9,884 points of interest (POIs) in Seattle. "
        r"We compare an Evolving Graph Convolutional Network (EvolveGCN) variant, a Multivariate Time Series Graph Neural Network (MTGNN), and temporal benchmarks across several input windows and forecast horizons. "
        + ("With 28 days of input, MTGNN achieves the lowest mean absolute error (MAE) at each evaluated horizon. " if mtgnn_wins else "")
        +
        r"For POIs with sparse histories, we measure positive-demand recovery and false activation separately. "
        r"We compare alternative adjacency matrices and perturb selected neighbor histories to examine when graph transfer reduces forecast error. "
        + (
            r"We use Gradient \(\times\) Input to measure feature-group shares and classify POIs by the source with the highest standardized attribution share. "
            if named else
            r"Gradient \(\times\) Input attribution identifies five groups with distinct profiles of reliance on historical visits and static features. "
        )
        + r"The same analysis quantifies the contributions of parent-place information and semantic descriptors at different levels of detail."
    )
    text = replace_between(text, r"\begin{abstract}", r"\end{abstract}", r"\begin{abstract}"+"\n"+headline+"\n")
    text = replace_between(text, "The results show that EvolveGCN with identity adjacency matrix",
                           r"\section{Literature Review}",
                           "The experiments compare error, demand recovery, and false activation on the same cohort. "
                           "The attribution analysis then relates the information used by the fitted model to POI characteristics.\n\n"
                           r"Section~\ref{sec:literature_review} reviews related work. Section~\ref{problem_setting} defines the dataset and tasks, "
                           r"Section~\ref{sec:methodology} presents the models and evaluation, and Section~\ref{sec:results} reports the results."
                           "\n\n")
    text = replace_once(text, r"\section{Literature Review}",
                        r"\section{Literature Review}"+"\n"+r"\label{sec:literature_review}")
    text = replace_between(text, "To capture both temporal visit patterns", "Geographic information is represented",
                           r"We use SafeGraph daily visit counts for \(N=9{,}884\) POIs in Seattle from October 1 through December 31, 2020. "
                           r"The 92-day panel contains 905,918 observed POI--days. For 110 POIs, all daily observations in October are missing. "
                           r"Each POI has static attributes for geographic location, functional identity, parent-place context, and opening hours."
                           "\n\n")
    text = replace_once(text, r"\caption{Distribution of mean observed visits across POI categories.}",
                        r"\caption{POI-level mean daily visits by category. Dots mark medians. Thick and thin lines span the 25th--75th and 5th--95th percentiles.}")
    text = replace_between(text, r"Figure~\ref{fig:visit_overview} summarizes",
                           r"\subsection{Prediction Task}",
                           r"Figure~\ref{fig:visit_overview} shows daily activity and how mean daily visits vary by POI category."
                           "\n\n"+f("fig:poi_sparsity_distribution")+"\n\n"+
                           r"In Figure~\ref{fig:poi_sparsity_distribution}, zero visits account for \(38.8\%\) of observed POI--days. "
                           r"The median daily count is one visit, and the 90th, 95th, and 99th percentiles are 6, 9, and 24. "
                           r"Across POIs, the median and first quartile of the nonzero-day share are \(66.3\%\) and \(35.9\%\), respectively. "
                           r"The variation motivates a separate evaluation of POIs with sparse visit histories."
                           "\n\n")
    text = replace_once(text, "We vary \\(T\\) and \\(H\\) separately",
                        r"Training targets end on November 26. Validation targets span November 27 through December 3, and test targets span December 4 through December 31. "
                        r"Training and validation windows contribute only eligible leads within their respective date blocks. "
                        r"We use zero placeholders for missing historical input entries and exclude missing targets from training and evaluation. "
                        r"Test windows start in the test block and cover their full horizon. "
                        r"The 28-day horizon therefore has one test forecast origin."
                        "\n\n"+r"We vary \(T\) and \(H\) separately")
    text = text.replace(" or correlations updated over rolling historical windows", "")
    text = replace_once(text, r"\subsection{Forecasting Models}",
                        r"Figure~\ref{fig:exp_logic} shows the two configurations used for the information-sharing and attribution analyses."
                        "\n\n"+f("fig:exp_logic")+"\n\n"+r"\subsection{Forecasting Models}")
    text = replace_between(text, "The historical-average predictor uses", r"\textit{(2) AR-ridge.}",
                           r"The historical-average (HA) forecast \(\bar y_j^{\mathrm{hist}}\) averages the observed input counts of POI \(j\) over the training windows. "
                           r"Every forecast step uses this value. Let \(\mathcal T_{\mathrm{tr}}\) denote the set of training dates and \(y_{\tau,j}\) the observed count for POI \(j\) on date \(\tau\)."
                           "\n\n")
    text = replace_once(text, r"\(\mathbf y_{b,j}=B_j^\top\mathbf v_{b,j}+\boldsymbol{\varepsilon}_{b,j}\),",
                        r"\(\mathbf y_{b,j}=\mathbf c_j+B_j^\top\mathbf v_{b,j}+\boldsymbol{\varepsilon}_{b,j}\),")
    text = replace_once(text, r"where \(B_j\in\mathbb{R}^{T\times H}\) is the coefficient matrix",
                        r"where \(\mathbf c_j\in\mathbb{R}^{H}\) is the intercept, \(B_j\in\mathbb{R}^{T\times H}\) is the coefficient matrix")
    text = replace_between(text, "EvolveGCN is designed for temporal graph data", r"\subsubsection{MTGNN}",
                           r"EvolveGCN models temporal graph data through recurrent graph updates \citep{pareja2020evolvegcn}. "
                           r"We use the repository graph-recurrent forecasting variant. At each input step, the model propagates visits through the normalized adjacency matrix and updates node states with a gated recurrent unit (GRU). "
                           r"A linear head maps the final node state and a skip connection from the last observed count to an \(H\)-step forecast."
                           "\n\n")
    text = replace_between(text, "MTGNN combines temporal convolution", r"\subsection{Adjacency Matrices}",
                           r"MTGNN combines temporal convolutions and multihop graph propagation \citep{wu2020connecting}. "
                           r"We use the fixed-adjacency configuration to compare predefined graphs. "
                           r"The MTGNN loss uses all observed training targets, including zero visits."
                           "\n\n")
    text = text.replace("using top category, sub-category, NAICS-2, and NAICS-4", "using top category, NAICS-2, and NAICS-4")
    text = text.replace(r"\mathrm{cat},\mathrm{subcat},\mathrm{naics2},\mathrm{naics4}",
                        r"\mathrm{cat},\mathrm{naics2},\mathrm{naics4}")
    text = replace_once(text, "Top category and NAICS-2 represent broader functional groups, whereas sub-category and NAICS-4 represent more specific classifications.",
                        "NAICS-2 represents broad industries. Top category and NAICS-4 use finer classifications.")
    text = replace_between(text, "To capture changes in visit-pattern similarity", r"\subsection{Forecast Evaluation}",
                           r"We use \(\eta_c=0.3\) and compute correlations from the training dates observed for both POIs."
                           "\n\n")
    text = replace_once(text, "All metrics are computed across test samples and POIs for each forecast horizon.",
                        "For each setting, we pool the observed test targets across forecast origins, leads, and POIs.")
    text = replace_once(text, "MAE and RMSE measure prediction errors, while RSE and RAE normalize these errors relative to the mean observed visits at each forecast horizon.",
                        "MAE and RMSE measure prediction errors. RSE and RAE scale the errors by the dispersion of pooled observed visits around their mean.")
    text = replace_between(text, r"\textit{(2) Visit-Density Sensitivity.}", r"\textit{(3) Low-History POI Evaluation.}",
                           r"\textit{(2) Visit-Density Sensitivity.}"+"\n"+
                           r"Let \(\mathcal T_{\mathrm{tr},j}\) be the dates with an observed count for POI \(j\) in the training period. "
                           r"The nonzero ratio is \(r_j=|\mathcal T_{\mathrm{tr},j}|^{-1}\sum_{\tau\in\mathcal T_{\mathrm{tr},j}}\mathbb{I}\{y_{\tau,j}>0\}\). "
                           r"The recent-density rule computes the same ratio over the final 14 training dates. The persistence-density rule multiplies \(r_j\) by the frequency of a positive count on the day after an observed positive count, using consecutive pairs with both counts observed. "
                           r"For each rule, we rank POIs by the score, retain those above the specified rank cutoff, and refit the model on the resulting panel."
                           "\n\n")
    text = replace_once(text, r"where \(z_j=1-r_j\) denotes the training-period zero-visit ratio, and \(Q_{\alpha}\) is the \(\alpha\)-quantile of positive historical averages across POIs.",
                        r"where \(z_j\) is the fraction of zero visits among observed training-window inputs and \(Q_{\alpha}\) is the \(\alpha\)-quantile of \(\bar y_j^{\mathrm{hist}}\) among POIs with positive historical means. We use \(\eta_z=0.70\) and \(\alpha=0.25\). Forecasts are clipped at zero before scoring.")
    text = replace_once(text, r"for POIs with nonempty neighborhoods.",
                        r"for POIs with nonempty neighborhoods. We set neighbor evidence to zero for an empty neighborhood.")
    text = replace_once(text, r"Table~\ref{tab:graph_transfer_regime_definitions} summarizes the resulting graph-transfer regimes.",
                        r"Table~\ref{tab:graph_transfer_regime_definitions} summarizes the regimes. We set \(\eta_G=0.1\) and \(\eta_{\mathrm{FAR}}=0.5\). "
                        r"We assign spurious, substitutive, confirmatory, redundant, and conservative regimes in that order. The remaining events form the Other group.")
    text = replace_between(text, "We consider several neighbor groups", r"\subsection{Feature Attribution and POI Clustering}",
                           r"Recent-active neighbors recorded positive visits in the last seven input days. "
                           r"High- and low-mean neighbors are above and below the median input mean among the neighbors. If no neighbor exceeds the median, the high-mean group includes those at the median. "
                           r"Top-10 active and correlated groups rank by the input mean and Pearson correlation, respectively. Random-10 draws the same number of neighbors. "
                           r"Aligned neighbors share at least one positive-visit day with the target. We also perturb the full neighborhood."
                           "\n\n")
    text = replace_once(text, "History includes historical visit counts; location includes latitude and longitude; identity includes category, industry, and brand information; parent includes parent-place information; and others includes opening-hour features and their missingness indicators.",
                        "History comprises visit counts. Location comprises latitude and longitude. Identity comprises top category, sub-category, NAICS-2, NAICS-4, and brand. Parent denotes parent-place information. Others contains opening hours and the opening-hour missingness indicator.")
    if named:
        text = text.replace("Feature Attribution and POI Clustering", "Feature Attribution and POI Regimes")
        text = replace_once(text,
                            "We then cluster POIs according to their attribution profiles to identify groups that rely on different information sources.",
                            "We then classify POIs by the source whose attribution share is highest relative to its distribution across POIs.")
        text = replace_once(text,
                            "cluster POIs according to their attribution profiles",
                            "classify POIs by their highest standardized attribution share")
        text = replace_between(
            text, r"\textit{(2) Attribution-Based POI Clustering.}", r"\textit{(3) Parent-Place Attribution.}",
            r"""\textit{(2) Attribution-Based POI Regimes.}
We standardize the shares in Equation~\ref{eq:attribution_group_share} across all \(N\) POIs separately for each feature group.
Let \(\bar{\pi}_g\) and \(s_g\) denote the mean and population standard deviation of the shares for group \(g\).
For a group with \(s_g>0\), the standardized share is
\begin{equation}
z_{j,g}=\frac{\pi_{j,g}-\bar{\pi}_g}{s_g}.
\label{eq:attribution_standardized_share}
\end{equation}
A constant share has \(z_{j,g}=0\).
We assign POI \(j\) to the regime defined by the group with the largest \(z_{j,g}\).
Ties follow the order history, location, identity, parent, and others.
The assigned regime identifies the source with the highest attribution relative to the variation across POIs for that source.
We summarize the original attribution shares and the forecasting error within each regime.

""",
        )
    else:
        text = replace_once(text, "We select the number of clusters based on silhouette scores and the interpretability of cluster centroids.",
                            r"We use \(K=5\) and fit \(K\)-means to the normalized shares.")
    text = replace_once(text, "Using MTGNN with identity adjacency, we compare the attribution of four semantic descriptors: top category, sub-category, NAICS-2, and NAICS-4.",
                        "Using the same GNN-static fit, we compare four semantic descriptors: top category, sub-category, NAICS-2, and NAICS-4.")
    text = text[:text.index(start)]
    text = text.replace(r"\usepackage{lscape,pdflscape}", r"\usepackage{lscape}")
    text = text.replace(r"\usepackage{float,adjustbox}", r"\usepackage{float}")
    text = text.replace(r"\usepackage{dsfont,physics}", r"\usepackage{dsfont}")
    text = replace_once(text, r"\begin{adjustbox}{max width=\textwidth}", r"\resizebox{\linewidth}{!}{")
    text = replace_once(text, r"\end{adjustbox}", "}")
    for old, new in {
        "from a POI's own past observations": "from the past observations at that POI",
        "each POI's visit history and attributes": "the visit history and attributes of each POI",
        "a POI's own visit history": "the visit history of the target",
        "related POIs' historical visit information": "historical visits from related POIs",
        "the target POI's prediction": "the forecast for the target POI",
        "the target POI's own history": "the history of the target POI",
        "each POI's prediction relies on its own historical visits and static attributes rather than information from other POIs":
            "the forecast for each POI uses its history and static attributes",
        "a target POI's own historical visits and static attributes": "the historical visits and static attributes of the target POI",
        "a POI's own historical visits predict future activity": "local visit histories predict future activity",
        "each POI's own information": "the information about each target POI",
        "the target POI's average historical visits": "the mean historical count for the target POI",
        "the target POI's original prediction": "the original forecast for the target POI",
    }.items():
        text = text.replace(old, new)

    def section(title, label, level):
        return "\\"+level+"{"+title+"}\n"+r"\label{"+label+"}"

    evol = mh.loc[(mh["T"]==28) & mh.series.eq("EvolveGCN (identity)")].sort_values("H")
    s7, s14, s28 = [number(v,2)+r"\%" for v in evol.skill_percent]
    ident = rec.loc[rec.model.eq("EvolveGCN") & rec.adjacency.eq("Identity")].iloc[0]
    top = rec.loc[rec.model.eq("EvolveGCN") & rec.adjacency.eq("Top category")].iloc[0]
    naics = graph.loc[graph.adjacency.eq("NAICS-2")]
    sub = naics.loc[naics.regime.eq("Substitutive transfer")].iloc[0]
    spurious = naics.loc[naics.regime.eq("Spurious transfer")].iloc[0]
    all_neighbors = pert.loc[pert.neighbor_group.eq("All")].iloc[0]
    majority = clusters.loc[clusters.n_pois.idxmax()]
    if named:
        title = "POI Attribution Regimes"
        group_intro = (
            rf"Figure~\ref{{fig:attribution_fingerprint}} reports the mean standardized attribution shares within each regime. "
            rf"The {majority.display_label} regime contains the most POIs "
            rf"({majority.n_pois:,}, {percent(majority.n_pois / clusters.n_pois.sum())})."
        )
        group_distribution = (
            r"Table~\ref{tab:category_regime_mae} gives the share of POIs assigned to each regime within each broad category. "
            r"Figure~\ref{fig:attribution_regime_map} shows their spatial distribution in Seattle."
        )
    else:
        title = "POI Groups with Different Prediction Mechanisms"
        cs = ", ".join(f"{row.n_pois:,}" for row in clusters.itertuples())
        group_intro = (
            rf"Figure~\ref{{fig:attribution_fingerprint}} shows the mean profiles of the five clusters. Their sizes are {cs} POIs, in cluster order. "
            rf"The largest cluster contains {majority.n_pois:,} POIs and has {majority.dominant_source.lower()} as its largest mean attribution share."
        )
        group_distribution = (
            r"Table~\ref{tab:category_regime_mae} summarizes the composition of the five clusters by POI category. "
            r"Figure~\ref{fig:attribution_regime_map} compares local cluster shares and same-cluster neighbor shares with their mapped-cohort references."
        )
    all_parent = parent.loc[parent.group.eq("All embedded POIs")].iloc[0]
    most_sem = sem.loc[sem.dominance_rate.idxmax()]
    first_dense = dens.loc[dens.subset.eq("Full")].iloc[0]
    last_dense = dens.loc[dens.subset.eq("Persist-dense75")].iloc[0]

    blocks = [
        section("Empirical Results", "sec:results", "section"),
        section("Forecasting Performance", "subsec:forecasting_results", "subsection"),
        section("Overall Performance", "subsubsec:overall_performance", "subsubsection"),
        r"Table~\ref{tab:multihorizon} compares the models on test targets from the same calendar block. "
        r"The identity-adjacency configuration gives a within-architecture reference for each graph model. "
        r"We select alternative adjacencies using validation loss and report their test performance.",
        section("Input Windows and Forecast Horizons", "subsubsec:multi_horizon_results", "subsubsection"),
        r"We consider \((T,H)\in\{(14,7),(28,7),(28,14),(28,28)\}\). The first two settings vary the input window at a fixed horizon. The final three vary the horizon at \(T=28\).",
        tab("tab:multihorizon"), f("fig:forecast_horizon_skill"),
        rf"Figure~\ref{{fig:forecast_horizon_skill}} shows that identity EvolveGCN achieves MAE skill of \({s7}\), \({s14}\), and \({s28}\) at the 7-, 14-, and 28-day horizons. "
        + ("The validation-selected MTGNN has the highest skill at each horizon. " if mtgnn_wins else "")
        + "The longest horizon has one test origin, so its result describes one 28-day trajectory per POI.",
        section("Low-History POIs", "subsubsec:low_history_results", "subsubsection"),
        r"\paragraph{Density sensitivity.} We rank POIs by nonzero ratio, nonzero ratio in the last 14 training days, or the product of nonzero ratio and the probability of two consecutive positive days conditional on a positive first day. "
        r"The cutoffs drop the lowest 10, 25, 50, or 75 percent of each ranking. We refit identity EvolveGCN on each retained subset at \(T=28\), \(H=14\).",
        tab("tab:sublevel"),
        rf"In Table~\ref{{tab:sublevel}}, the nonzero ratio is \({number(first_dense.nonzero_ratio)}\) for the full panel and \({number(last_dense.nonzero_ratio)}\) for the densest persistence subset. Their respective MAE skills are \({number(first_dense.skill_percent,1)}\%\) and \({number(last_dense.skill_percent,1)}\%\).",
        r"\paragraph{Zero-demand collapse.}"+
        f" The low-history rule selects {diag['low_history_pois']:,} POIs at "+
        r"\(T=14\), \(H=7\). "+
        f"The test set contains {diag['low_history_events']:,} events for these POIs, including {diag['positive_low_history_events']:,} positive events.",
        tab("tab:zero_collapse_diagnostic"), f("fig:recovery_activation_tradeoff"),
        rf"Figure~\ref{{fig:recovery_activation_tradeoff}} compares the amount of realized positive demand recovered with the share of zero-demand events activated. Identity EvolveGCN yields a DMR of \({percent(ident.dmr_model)}\) and a FAR of \({percent(ident.far_0_5)}\). The top-category configuration yields \({percent(top.dmr_model)}\) and \({percent(top.far_0_5)}\), respectively.",
        section("Information Sharing among POIs", "subsec:information_sharing_results", "subsection"),
        section("Alternative Adjacency Matrices", "subsubsec:adjacency_results", "subsubsection"),
        rf"At \(T=28\), \(H=14\), we compare the visit-only EvolveGCN fits under functional and identity adjacency. The low-history subset contains {graph_stats['low_history_pois']:,} POIs and {graph_stats['n_events']:,} test events.",
        tab("tab:graph_transfer_summary"), f("fig:counterfactual_graph_transfer_map"),
        rf"Table~\ref{{tab:graph_transfer_summary}} identifies {sub.n_events:,} substitutive and {spurious.n_events:,} spurious events under NAICS-2. "
        rf"Figure~\ref{{fig:counterfactual_graph_transfer_map}} conditions the same comparison on observed demand. Average gain is \({number(graph_stats['mean_gain_positive'],3)}\) visits for positive-demand events and \({number(graph_stats['mean_gain_zero'],3)}\) for zero-demand events.",
        tab("tab:semantic_granularity_transfer"),
        r"Table~\ref{tab:semantic_granularity_transfer} compares the substitutive and spurious regimes across functional graphs."
        + (" Top category and NAICS-4 form identical neighbor sets in this cohort and produce the same results." if naics_equal else ""),
        section("Related-POI Perturbation", "subsubsec:perturbation_results", "subsubsection"),
        r"Under NAICS-4, a recovered positive event has at least 0.5 predicted visits under the graph model, less than 0.5 under identity, and lower absolute error under the graph model. "
        r"We measure recovery loss and false activation removal when the perturbed prediction drops below 0.5 visits in the respective reference sets.",
        tab("tab:metric_targeted_perturbation_naics4"), f("fig:neighbor_perturbation_tradeoff"),
        rf"Figure~\ref{{fig:neighbor_perturbation_tradeoff}} compares {perturb_stats['harmful_false_activations']:,} false activations and {perturb_stats['recovered_positives']:,} recovered positives among attempted substitutions. Zeroing all neighbors removes \({percent(all_neighbors.false_activations_removed)}\) of false activations and \({percent(all_neighbors.recoveries_lost)}\) of recovered positives.",
        section("Explaining POI Visit Predictions", "subsec:explanation_results", "subsection"),
        section("Attribution Patterns", "subsubsec:attribution_patterns", "subsubsection"),
        r"We compute Gradient \(\times\) Input for the GNN-static fit at \(T=14\), \(H=7\). The calculation uses untruncated model outputs, standardized visit counts, and encoded static attributes. Each POI profile summarizes the five feature-group shares in Equation~\ref{eq:attribution_group_share}.",
        section(title, "subsubsec:attribution_clusters", "subsubsection"),
        group_intro,
        f("fig:attribution_fingerprint"), tab("tab:attri_regime"), tab("tab:category_regime_mae"),
        group_distribution,
        f("fig:attribution_regime_map"),
        section("Parent-Place Information", "subsubsec:parent_results", "subsubsection"),
        r"We examine parent-place attribution for POIs with a recorded parent. The combined own share includes visit history, location, identity, and opening-hour features.",
        tab("tab_parent"),
        rf"Across {all_parent.n_pois:,} embedded POIs in Table~\ref{{tab_parent}}, the parent-place feature group accounts for \({percent(all_parent.parent_share)}\) of the mean attribution share. The remaining information sources account for \({percent(all_parent.own_share)}\).",
        section("Category and Industry Granularity", "subsubsec:granularity_results", "subsubsection"),
        r"We compare the top-category, sub-category, NAICS-2, and NAICS-4 features in the same identity-adjacency GNN-static fit. "
        r"The mean share uses the full attribution total per event. Dominance identifies the largest of the four semantic attributions.",
        tab("tab:semantic_granularity"),
        rf"In Table~\ref{{tab:semantic_granularity}}, {most_sem.descriptor} has the highest dominance rate at \({percent(most_sem.dominance_rate,2)}\). The dominant-event MAE measures error on the events assigned to each descriptor.",
        r"\bibliographystyle{abbrvnat}", r"\bibliography{reference}", r"\end{document}",
    ]
    text = text.rstrip()+"\n\n"+"\n\n".join(blocks)+"\n"
    text = text.replace("(P,H)", "(T,H)")
    return text, tables, figures


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--manuscript", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    body, tables, figures = build_front(args.manuscript, args.data, args.output/"art", args.panel)
    labels = re.findall(r"\\label\{([^}]+)\}", body)
    refs = set(re.findall(r"\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}", body))
    assert len(labels) == len(set(labels)) and refs <= set(labels), refs-set(labels)
    assert r"\section{Feature attribution for neural networks}" not in body
    (args.output/"manuscript_front_revised.tex").write_text(body, encoding="utf-8")
    blocks = [f"% {position}\n{figures[label]}" for _, _, label, position, _ in FIGURES]
    (args.output/"figure_blocks.tex").write_text("\n\n".join(blocks)+"\n", encoding="utf-8")
    folder = args.output/"tables"
    folder.mkdir(exist_ok=True)
    for label, table_body in tables.items():
        (folder/(label.replace(":", "_")+".tex")).write_text(table_body+"\n", encoding="utf-8")
    print(f"Assembled {len(tables)} recomputed tables and {len(figures)+1} manuscript figures.")


if __name__ == "__main__":
    main()
