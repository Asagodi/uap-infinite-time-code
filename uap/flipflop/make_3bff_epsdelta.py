#!/usr/bin/env python3
"""Single delta(eps) panel for the CtD 3-bit flip-flop, for main-text Figure 4(D):
the empirical eps-delta curve of the independent Neural ODE vs the trained GRU,
scored on the network's own readout. Loads the saved distilled model. Run in CtDEnv."""
import os, numpy as np, torch
from uap.flipflop import fit_node_tube as F

FIGDIR, MODELDIR = F.FIGDIR, F.MODELDIR
f = F.Field(384, 3)
f.load_state_dict(torch.load(os.path.join(MODELDIR, "node_ctd_3bff_tube.pt")))
f.eval()
ev = F.evaluate(f)
sup = ev['sup']
print(f"3bff: bit-acc {ev['bit']*100:.2f}%  median-sup {np.median(sup):.3f}  delta(0.3)={(sup>0.3).mean():.3f}", flush=True)

import matplotlib.font_manager as fm, matplotlib.pyplot as plt
for _f in fm.findSystemFonts(fontpaths=[os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")]):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 26, "axes.labelsize": 32, "xtick.labelsize": 26, "ytick.labelsize": 26,
    "axes.linewidth": 1.4, "lines.linewidth": 3.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "savefig.dpi": 300})
print("font:", os.path.basename(fm.findfont(fm.FontProperties(family="serif"))), flush=True)

fig, ax = plt.subplots(figsize=(5.0, 4.6))
eps = np.logspace(-4, np.log10(3), 90)
ax.semilogx(eps, [(sup > e).mean() for e in eps], color="#0072B2", lw=3.4)
ax.axvline(np.median(sup), color="0.55", lw=2.2, ls="--")
ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel(r"$\delta(\varepsilon)$")
ax.set_xlim(1e-4, 3); ax.set_ylim(-0.03, 1.03)
ax.set_xticks([1e-4, 1e-2, 1e0]); ax.set_yticks([0, 0.5, 1.0])
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "node_3bff_epsdelta.pdf"))
fig.savefig(os.path.join(FIGDIR, "node_3bff_epsdelta.png"))
print("saved node_3bff_epsdelta", flush=True)
