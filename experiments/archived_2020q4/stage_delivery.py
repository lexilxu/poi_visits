"""Assemble the formal experiment directory and compact manuscript package."""
from argparse import ArgumentParser
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pypdf

from assemble_formal_paper import ATTRIBUTION_REGIMES, ATTRIBUTION_SOURCES, FIGURES


DATA_FILES = {
    "exp1": ["07_horizon_skill.csv", "multi_horizon_results.csv", "forecast_protocol.json", "model_fit_summaries.json"],
    "exp2": ["01_recovery_activation.csv", "density_sensitivity.csv"],
    "exp3": ["02_graph_transfer_cells.csv", "02_graph_transfer_regimes.csv", "graph_transfer_summary.json"],
    "exp4": ["03_neighbor_perturbation.csv", "perturbation_summary.json"],
    "exp5": ["04_attribution_clusters.csv", "04_attribution_fingerprint.csv", "attribution_cluster_assignments.csv",
             "attribution_summary.json", "category_cluster_distribution.csv"],
    "exp6": ["parent_attribution.csv", "semantic_attribution.csv"],
}
TABLE_FILES = {
    "tab_multihorizon": ("exp1", "table_multihorizon"),
    "tab_sublevel": ("exp2", "table_density_sensitivity"),
    "tab_zero_collapse_diagnostic": ("exp2", "table_zero_collapse_diagnostic"),
    "tab_graph_transfer_summary": ("exp3", "table_graph_transfer_summary"),
    "tab_semantic_granularity_transfer": ("exp3", "table_semantic_granularity_transfer"),
    "tab_metric_targeted_perturbation_naics4": ("exp4", "table_metric_targeted_perturbation_naics4"),
    "tab_attri_regime": ("exp5", "table_attri_regime"),
    "tab_category_regime_mae": ("exp5", "table_category_regime_mae"),
    "tab_parent": ("exp6", "table_parent_place_attribution"),
    "tab_semantic_granularity": ("exp6", "table_semantic_granularity"),
}
LEGACY_APPENDIX_ART = "legacy/low_history_graph_transfer_two_panel_map.png"


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert digest(source) == digest(target)


def write(path, text):
    path.write_text(text.rstrip()+"\n", encoding="utf-8")


def make_guide(package, data, named):
    protocol = json.loads((data/"forecast_protocol.json").read_text(encoding="utf-8"))
    r = protocol["T14_H7"]
    g = json.loads((data/"graph_transfer_summary.json").read_text(encoding="utf-8"))["NAICS-2"]
    regime_title = "POI Attribution Regimes" if named else "POI Groups with Different Prediction Mechanisms"
    regime_caption = (
        "每个 POI 按标准分数最高的来源分组。热图展示各组归因标准分数的均值，表格和地图沿用同一分组。"
        if named else "热图、归因组表和类别分布表用同一批聚类及真实归因份额"
    )
    guide = f"""# 正式实验插图与正文替换

把 `art/` 整个放到论文工程根目录。`manuscript_front_revised.tex` 是可以直接对照的主文改稿，含全部图、表、交叉引用和替换后的英文正文。`figure_blocks.tex` 和 `tables/` 可用于逐块合并。参考文献沿用论文工程里的 `reference.bib`。

以下行号指这次贴出的 LaTeX。新版主文止于原稿第 1616 行，第 1617 行 `Feature attribution for neural networks` 是旧版重复章节的起点。图表通过章节和标签定位，排版后的页码会随浮动体和参考文献调整。

| PDF | 原稿插入位置 | 正文或表格的配套修改 |
|---|---|---|
| `daily_visit.pdf`、`category_visit.pdf` | Dataset and POI Attributes，原 372–391 行，保留 `fig:visit_overview` 双子图 | 图注说明类别图的中位数、四分位区间和 5–95 百分位区间 |
| `poi_sparsity_distribution.pdf` | 访问概览图之后、Prediction Task（407 行）之前 | 用实测统计替换 392–405 行。零访问 38.8%，非零天数占比的中位数和第一四分位为 66.3% 和 35.9% |
| `exp_logic.pdf` | Methodology 开头、Forecasting Models（474 行）之前 | 把原 784–789 行的同名图块移到这里，沿用 `fig:exp_logic` |
| `forecasting_performance_across_horizons.pdf` | Input Windows and Forecast Horizons，`tab:multihorizon`（886–887 行）之后 | 用新预测表和技能曲线，固定 \\(T=28\\)，写明 \\(H=28\\) 只有一个测试起点 |
| `recovery_activation_tradeoff.pdf` | Zero-demand collapse，`tab:zero_collapse_diagnostic` 结束（1002 行）之后 | 换成 {r['low_history_pois']:,} 个低历史 POI、{r['low_history_events']:,} 个测试事件的诊断表和正文 |
| `outcome_conditioned_graph_transfer.pdf` | Alternative Adjacency Matrices，更新后的 `tab:graph_transfer_summary`（1103 行）之后 | 替换 1124–1129 行旧 PNG 图块，沿用图标签。新分组有 {g['low_history_pois']:,} 个低历史 POI，正需求和零需求平均 MAE gain 分别为 {g['mean_gain_positive']:+.3f} 和 {g['mean_gain_zero']:+.3f} |
| `neighbor_perturbation_tradeoff.pdf` | Related-POI Perturbation，`tab:metric_targeted_perturbation_naics4`（1260 行结束）之后 | 更新受扰动邻居组的对照表、分母和结果解释 |
| `attribution_fingerprint.pdf` | {regime_title}，原 1318 行的 `tab:attri_regime` 之前 | {regime_caption} |
| `attribution_cluster_map.pdf` | 同一小节，类别分布表之后、Parent-Place Information（1464 行）之前 | 替换原 1452–1457 行空间图，保留 `fig:attribution_regime_map` |

## 实验口径

- 访问描述、预测、图迁移、邻居扰动和归因都从 2020-10-01 至 2020-12-31 的 9,884 个 POI 面板出发。密度敏感性使用该面板的训练期活动度子集。
- 110 个 POI 的 3,410 条十月缺失记录在源面板中保留缺失标记。模型以零作为缺失历史输入的占位值，训练和评分只使用观测到的目标。真实的零访问计入损失和指标。
- 训练目标截至 11 月 26 日，验证目标为 11 月 27 日至 12 月 3 日，测试目标为 12 月 4 日至 31 日。训练和验证仅计入各自区间的预测步。测试采用完整窗口。28 天预测期有一个测试起点。
- 正文统一用 \\(T\\) 表示历史窗口，\\(H\\) 表示预测期。图模型比较使用固定邻接。类别、行业粒度和父地点归因均来自同一个含 sub-category 特征的 GNN-static 拟合。
"""
    write(package/"insertion_guide.md", guide)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--descriptive", type=Path, required=True)
    parser.add_argument("--experiment-output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    package, data, exp = args.package, args.data, args.experiment_output
    exp.mkdir(parents=True, exist_ok=True)
    manuscript = (package/"manuscript_front_revised.tex").read_text(encoding="utf-8")
    manifest = json.loads((data/"dataset_manifest.json").read_text(encoding="utf-8"))
    assert (manifest["date_start"], manifest["date_end"], manifest["nodes"]) == ("2020-10-01", "2020-12-31", 9884)
    fit_records = json.loads((data/"model_fit_summaries.json").read_text(encoding="utf-8"))
    assert len(fit_records) == 57
    ids = pd.read_csv(data/"attribution_cluster_assignments.csv")
    sizes = pd.read_csv(data/"04_attribution_clusters.csv")
    assert len(ids) == ids.placekey.nunique() == int(sizes.n_pois.sum()) == 9884
    counts = ids.cluster.value_counts()
    assert set(counts.index) <= set(sizes.display_label)
    assert counts.reindex(sizes.display_label, fill_value=0).tolist() == sizes.n_pois.tolist()
    named = tuple(sizes.display_label) == ATTRIBUTION_REGIMES
    if named:
        shares = ids[list(ATTRIBUTION_SOURCES)].to_numpy()
        scores = ids[[f"{group}_z" for group in ATTRIBUTION_SOURCES]].to_numpy()
        assert np.isfinite(shares).all() and (shares >= 0).all() and np.allclose(shares.sum(axis=1), 1)
        centered = shares - shares.mean(axis=0)
        spread = shares.std(axis=0)
        expected_scores = np.divide(centered, spread, out=np.zeros_like(centered), where=spread > 0)
        assert np.allclose(scores, expected_scores)
        assert np.array_equal(ids.cluster, np.asarray(ATTRIBUTION_REGIMES)[scores.argmax(axis=1)])

    file_map = {name: folder for folder, name, *_ in FIGURES} | {"daily_visit.pdf":"exp0", "category_visit.pdf":"exp0"}
    art_names = re.findall(r"\\includegraphics\[[^\]]+\]\{art/([^}]+)\}", manuscript)
    latest = package/"manuscript_attribution_regimes.tex"
    reviewed = latest.read_text(encoding="utf-8") if latest.is_file() else manuscript
    active_tex = re.sub(r"(?<!\\)%.*", "", reviewed)
    art_refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", active_tex)
    legacy_art = package/"art"/LEGACY_APPENDIX_ART
    assert art_refs.count("art/"+LEGACY_APPENDIX_ART) == int(legacy_art.is_file())
    missing = [name for name in art_refs if not (package/name).is_file()]
    assert not missing, missing
    assert set(art_names) == set(file_map) and len(art_names) == len(file_map) == 10
    assert {p.name for p in (package/"art").glob("*") if p.is_file()} == set(file_map)
    figure_records = []
    configure = getattr(pypdf, "apply_configuration", None)
    # The Seattle vector map exceeds the default two-million-token stream budget.
    with configure(stream_parsing_work_maximum_count=4_000_000) if configure else nullcontext():
        for name in art_names:
            dest = exp/file_map[name]/name
            src = package/"art"/name
            pdf = pypdf.PdfReader(src)
            assert len(pdf.pages) == 1 and not pdf.pages[0].images
            copy(src, dest)
            page = pdf.pages[0]
            figure_records.append({
                "path": dest.relative_to(exp).as_posix(), "sha256": digest(dest), "vector": True,
                "width_mm": round(float(page.mediabox.width)/72*25.4, 2),
                "height_mm": round(float(page.mediabox.height)/72*25.4, 2),
            })
    for folder, files in DATA_FILES.items():
        for name in files:
            copy(data/name, exp/folder/name)
    for stem, (folder, name) in TABLE_FILES.items():
        copy(package/"tables"/f"{stem}.tex", exp/folder/f"{name}.tex")
    for name in ("06_daily_count_distribution.csv", "06_poi_nonzero_ratio_distribution.csv", "table_poi_attributes.tex"):
        copy(args.descriptive/name, exp/"exp0"/name)
    write(exp/"result_manifest.json", json.dumps(
        {"dataset": manifest, "figures": figure_records, "fit_records": len(fit_records)},
        ensure_ascii=False, indent=2))
    insertions = (package/"figure_blocks.tex").read_text(encoding="utf-8")
    for name, folder in file_map.items():
        insertions = insertions.replace("{art/"+name+"}", "{exp/"+folder+"/"+name+"}")
    write(exp/"figure_insertions.tex", insertions)
    make_guide(package, data, named)
    copy(package/"insertion_guide.md", exp/"README.md")
    with ZipFile(args.zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(package.rglob("*")):
            if file.is_file():
                relative = file.relative_to(package).as_posix()
                assert file.suffix in {".pdf", ".tex", ".md"} or relative == "art/"+LEGACY_APPENDIX_ART
                archive.write(file, relative)
    with ZipFile(args.zip) as archive:
        assert archive.testzip() is None
        assert sum(name.endswith(".pdf") for name in archive.namelist()) == 10
    groups = "regime" if named else "cluster"
    print(f"Verified {len(art_names)} PDFs, {len(TABLE_FILES)} recomputed tables, and 9,884 {groups} assignments.")


if __name__ == "__main__":
    main()
