"""Regenerate two archived plots with the audited architecture name."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/archived_2020q4'))
import plot_formal
base = plot_formal.base
base.DATA = ROOT/'artifacts/archived_2020q4/figure_data'
base.ART = ROOT/'paper/revision'
base.style()
original_output = base.output


def output(fig, key, title):
    for artist in fig.findobj(Text):
        artist.set_text(artist.get_text().replace('EvolveGCN','GraphGRU'))
    original_output(fig,key,title)


base.output = output
plot_formal.recovery()
plot_formal.horizons()
print('Regenerated recovery and horizon figures from archived aggregate values.')
