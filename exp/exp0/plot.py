"""Render the final daily and category plots at their original 8.8 by 5.8 inch size."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from exp.common.paths import RESULTS_ROOT

BLUE, GREY = '#245A7C', '#7E8990'


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=RESULTS_ROOT/'exp0/data')
    p.add_argument('--out', type=Path, default=RESULTS_ROOT/'exp0/art')
    p.add_argument('--dpi', type=int, default=600)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    figures = []
    daily = pd.read_csv(a.data/'daily_mean_visits.csv', parse_dates=['date'])
    fig, ax = plt.subplots(figsize=(8.8, 5.8), layout='constrained')
    ax.plot(daily.date, daily.mean_observed_visits, color=GREY, marker='o', markersize=3, linewidth=1.3, label='Daily mean')
    ax.plot(daily.date, daily.centered_7_day_mean, color=BLUE, linewidth=2.6, label='Centered 7-day mean')
    ax.set(xlim=(daily.date.iloc[0], daily.date.iloc[-1]), ylim=(0, 4), xlabel='Date (2020)')
    ax.set_ylabel('Mean observed visits per POI', labelpad=14)
    ax.set_xticks(pd.to_datetime(['2020-10-01', '2020-10-15', '2020-11-01', '2020-11-15', '2020-12-01', '2020-12-15', '2020-12-31']))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%d'))
    ax.legend(loc='upper right', frameon=False, ncol=2)
    figures.append(('daily_visit', fig))
    category = pd.read_csv(a.data/'category_quantiles.csv')
    table = category.loc[category.measure.eq('mean_observed_visits')].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), layout='constrained')
    y = np.arange(len(table))
    ax.hlines(y, table.q5, table.q95, color=GREY, linewidth=1.5)
    ax.hlines(y, table.q25, table.q75, color=BLUE, linewidth=4.5)
    ax.scatter(table.q50, y, color=BLUE, s=55, edgecolor='white', zorder=4)
    labels = ['\n'.join(textwrap.wrap(c, width=35, break_long_words=False, break_on_hyphens=False)) for c in table.category]
    ax.set_yticks(y, labels, fontsize=13)
    ax.set_ylim(len(table)-.3, -.7)
    ax.set_xlabel('Mean observed visits per day')
    ax.set_xscale('function', functions=(np.log1p, np.expm1))
    ax.set_xticks([0, 1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
    ax.set_xlim(0, 20)
    for i, row in table.iterrows():
        ax.annotate(f'n = {row.pois:,}', xy=(row.q95, i), xytext=(7, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=13, fontweight='bold')
    figures.append(('category_visit', fig))
    for name, fig in figures:
        for suffix in ('pdf', 'svg', 'png'):
            fig.savefig(a.out/f'{name}.{suffix}', dpi=a.dpi)
        plt.close(fig)
    record = dict(canvas_inches=[8.8, 5.8], raster_dpi=a.dpi,
                  input_sha256={n: sha256((a.data/n).read_bytes()).hexdigest() for n in
                                ['daily_mean_visits.csv', 'category_quantiles.csv']})
    (a.out/'descriptive_figures_audit.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(a.out)


if __name__ == '__main__':
    main()
