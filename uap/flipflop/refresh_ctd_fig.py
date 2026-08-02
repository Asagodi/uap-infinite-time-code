#!/usr/bin/env python3
"""Rebuild the CtD NODE figure from the saved model (no retraining). Panel C now
shows only STABLE fixed points (Jacobian of the autonomous field has all eigenvalue
real parts < 0), so saddles between memory states are excluded and the 8 memory-state
attractors are clean. Run in CtDEnv."""
import os, numpy as np, torch
from uap.flipflop import fit_node_tube as F

FIGDIR, MODELDIR = F.FIGDIR, F.MODELDIR
H, U, readout, THR, rollout, Hf, N, NU, P = F.H, F.U, F.readout, F.THR, F.rollout, F.Hf, F.N, F.NU, F.P

f = F.Field(384, 3)
f.load_state_dict(torch.load(os.path.join(MODELDIR, "node_ctd_3bff_tube.pt")))
f.eval()
ev = F.evaluate(f); sup = ev['sup']; d03 = float((sup > 0.3).mean())


def stable_fps(f, n_seed=900, iters=3000):
    z = torch.zeros(1, NU)
    X = Hf[torch.randint(0, P, (n_seed,))].clone().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=2e-2)
    for _ in range(iters):
        opt.zero_grad(); g = f(X, z.expand(X.shape[0], NU))
        (g ** 2).sum(1).mean().backward(); opt.step()
    X = X.detach()
    with torch.no_grad():
        sp = f(X, z.expand(X.shape[0], NU)).norm(dim=1)
    Xk = X[sp < 1e-4]                                         # genuine fixed points
    keep = []
    for i in range(Xk.shape[0]):
        xi = Xk[i:i + 1]
        J = torch.autograd.functional.jacobian(
            lambda x: f(x, z.expand(1, NU)).squeeze(0), xi, vectorize=True).reshape(N, N)
        keep.append(bool(torch.linalg.eigvals(J).real.max() < 0))   # continuous-flow stable
    Xs = Xk[torch.tensor(keep)] if len(keep) else Xk[:0]
    ro = readout(Xs).numpy()
    patt = np.unique((ro > THR).astype(int), axis=0) if len(ro) else np.zeros((0, 3))
    return len(patt), ro, len(Xs)


nfp, roF, nst = stable_fps(f)
print(f"model bit-acc={ev['bit']*100:.2f}% median-sup={np.median(sup):.3f} delta(0.3)={d03:.3f} "
      f"stable-FPs={nst} distinct-states={nfp}/8", flush=True)

# rollouts for overlay
with torch.no_grad():
    xs = rollout(f, H[:, 0], U)
yg = readout(H).numpy(); yn = readout(xs).numpy()
bit_g = (yg > THR).astype(int)
flips = np.abs(np.diff(bit_g, axis=1)).sum(axis=(1, 2))
trial = int(np.argsort(flips)[-1])

import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
for _f in fm.findSystemFonts(fontpaths=[os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")]):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
    "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"], "mathtext.fontset": "cm",
    "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 17, "xtick.labelsize": 15,
    "ytick.labelsize": 15, "legend.fontsize": 14, "axes.linewidth": 1.2, "lines.linewidth": 2.2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 140, "savefig.dpi": 300})
print("font:", os.path.basename(fm.findfont(fm.FontProperties(family="serif"))), flush=True)
OK = ["#0072B2", "#D55E00", "#009E73"]


def panel_labels(axes, letters="ABC", dy=8):
    fig = axes[0].figure; fig.canvas.draw(); r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted(); size = 17   # match the hero row (viz_8fp.py)
    for ax, L in zip(axes, letters):
        x0 = ax.get_tightbbox(r).transformed(inv).x0
        y = ax.get_position().y1 + dy / (fig.get_size_inches()[1] * fig.dpi)
        fig.text(x0, y, f"({L})", ha="left", va="bottom", fontsize=size, fontweight="bold")


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

eps = np.logspace(-2.3, 0.4, 70)
ax[0].plot(eps, [(sup > e).mean() for e in eps], color="#0072B2", lw=2.8)
ax[0].axvline(np.median(sup), color="0.55", lw=1.4, ls="--")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\varepsilon$  (readout sup-error)")
ax[0].set_ylabel(r"$\delta(\varepsilon)$"); ax[0].set_ylim(-0.02, 1.02)
ax[0].set_title(f"$\\varepsilon$-$\\delta$ closeness\nbit-accuracy {ev['bit']*100:.1f}%")

T = yg.shape[1]; tt = np.arange(T)
for b in range(3):
    off = b * 1.35
    ax[1].plot(tt, yg[trial, :, b] + off, color=OK[b], lw=2.8)
    ax[1].plot(tt, yn[trial, :, b] + off, color="k", lw=1.3, ls=(0, (3, 2)))
    ax[1].axhline(THR[b] + off, color="0.85", lw=0.9, zorder=0)
ax[1].xaxis.set_major_locator(MaxNLocator(4, integer=True))
ax[1].set_yticks([]); ax[1].set_xlabel("time step"); ax[1].set_ylabel("readout  (3 bits)")
ax[1].set_title("Memory tracked through flips")

c3 = np.where(roF[:, 2] > THR[2], "#0072B2", "#D55E00")
ax[2].scatter(roF[:, 0], roF[:, 1], c=c3, s=52, alpha=0.85, zorder=2, edgecolors="none")
ax[2].axvline(THR[0], color="0.85", lw=1); ax[2].axhline(THR[1], color="0.85", lw=1)
ax[2].set_xlabel("readout bit 1"); ax[2].set_ylabel("readout bit 2")
ax[2].set_title(f"NODE stable fixed points\n{nfp}/8 memory states")

fig.tight_layout(rect=[0, 0, 1, 0.9])
panel_labels(ax, "FGH")
fig.savefig(os.path.join(FIGDIR, "node_ctd_3bff_fit.pdf"))
fig.savefig(os.path.join(FIGDIR, "node_ctd_3bff_fit.png"))
print("saved node_ctd_3bff_fit.pdf/.png  trial=", trial, "flips=", int(flips[trial]), flush=True)
