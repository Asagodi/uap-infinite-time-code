#!/usr/bin/env python3
"""Figure 4(C): real vs fitted separatrices of the trained 3-bit flip-flop network.
On a 2D slice through the four memory-state fixed points that share bit 3, we roll
out both the trained network and the independent Neural ODE (zero input) from each
initial condition and record which memory state each reaches. The boundaries of the
trained network's basins (solid) and the Neural ODE's basins (dashed) are drawn on
top of the (faint) partition; where they coincide, the approximation preserves the
basins, exactly as the separatrices coincide in panel (A). Run in CtDEnv."""
import os, numpy as np, torch
from uap.flipflop import fit_node_tube as F

FIGDIR, MODELDIR = F.FIGDIR, F.MODELDIR
cell, readout, THR, N, NU = F.cell, F.readout, F.THR, F.N, F.NU
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#BBBBBB"]


def pidx(p): return int(p[0] * 4 + p[1] * 2 + p[2])


fp = np.load(os.path.join(MODELDIR, "ctd_3bff_fps.npz"))
xs, q, stab = fp["xs"], fp["q"], fp["stab"].astype(bool)
X = xs[(q < 1e-4) & stab]
ro = readout(torch.tensor(X, dtype=torch.float64)).numpy()
pat = (ro > THR).astype(int)
patterns = np.unique(pat, axis=0)
reps = np.stack([X[(pat == p).all(1)].mean(0) for p in patterns])
sel = np.array([i for i, p in enumerate(patterns) if p[2] == 0])
reps4 = reps[sel]; pats4 = patterns[sel]
mu = reps4.mean(0)
_, _, Vt = np.linalg.svd(reps4 - mu, full_matrices=False); PC = Vt[:2]
coords = (reps4 - mu) @ PC.T
span = coords.max(0) - coords.min(0)
lo = coords.min(0) - 0.4 * span - 0.5
hi = coords.max(0) + 0.4 * span + 0.5
ng = 170
ga = np.linspace(lo[0], hi[0], ng); gb = np.linspace(lo[1], hi[1], ng)
AA, BB = np.meshgrid(ga, gb)
grid = mu[None] + AA.ravel()[:, None] * PC[0][None] + BB.ravel()[:, None] * PC[1][None]

f_node = F.Field(384, 3)
f_node.load_state_dict(torch.load(os.path.join(MODELDIR, "node_ctd_3bff_tube.pt")))
f_node.eval()

M = grid.shape[0]; steps = 280
zt = torch.zeros(M, NU, dtype=torch.float64)


@torch.no_grad()
def basin_labels(step_fn):
    h = torch.tensor(grid, dtype=torch.float64)
    for _ in range(steps):
        h = step_fn(h)
    p = (readout(h).numpy() > THR).astype(int)
    return np.array([pidx(pp) for pp in p]).reshape(ng, ng)


labg = basin_labels(lambda h: cell(zt.float(), h.float()).to(torch.float64))       # trained network
labn = basin_labels(lambda h: h + f_node(h, zt))                                   # Neural ODE
print("network basins", sorted(np.unique(labg).tolist()), "| NODE basins", sorted(np.unique(labn).tolist()), flush=True)
print(f"basin agreement over slice: {(labg == labn).mean()*100:.2f}%", flush=True)

# fit (Neural ODE) fixed points: descend the Neural ODE field from each target FP
z1 = torch.zeros(1, NU)
Xn = torch.tensor(reps4, dtype=torch.float64).clone().requires_grad_(True)
optn = torch.optim.Adam([Xn], lr=8e-3)
for _ in range(4000):
    optn.zero_grad(); g = f_node(Xn, z1.expand(Xn.shape[0], NU))
    (g ** 2).sum(1).mean().backward(); optn.step()
node_xy = (Xn.detach().numpy() - mu) @ PC.T

import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
for _f in fm.findSystemFonts(fontpaths=[os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")]):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 24, "axes.labelsize": 30, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "axes.linewidth": 1.4, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "savefig.dpi": 300})
print("font:", os.path.basename(fm.findfont(fm.FontProperties(family="serif"))), flush=True)

fig, ax = plt.subplots(figsize=(5.2, 5.0))
# faint partition (trained network) for context
ax.pcolormesh(AA, BB, labg, cmap=ListedColormap(OKABE), norm=BoundaryNorm(np.arange(-0.5, 8.5, 1), 8),
              shading="auto", alpha=0.22)
# real (solid) and fitted (dashed) separatrices


def draw_boundaries(lab, **kw):
    for k in np.unique(lab):
        ax.contour(AA, BB, (lab == k).astype(float), levels=[0.5], **kw)


draw_boundaries(labg, colors="k", linewidths=3.2)
draw_boundaries(labn, colors="#D55E00", linewidths=2.6, linestyles="dashed")
fpc = [OKABE[pidx(p)] for p in pats4]                                                      # basin color per FP
ax.scatter(coords[:, 0], coords[:, 1], marker="o", c=fpc, s=230, edgecolors="k", lw=2.4, zorder=4)     # target FP
ax.scatter(node_xy[:, 0], node_xy[:, 1], marker="*", c=fpc, s=520, edgecolors="k", lw=1.6, zorder=5)   # fit FP
# legend proxies -> caption instead; keep panel clean
ax.set_xlabel("latent PC 1"); ax.set_ylabel("latent PC 2")
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "node_3bff_basins.pdf"))
fig.savefig(os.path.join(FIGDIR, "node_3bff_basins.png"))
print("saved node_3bff_basins", flush=True)
