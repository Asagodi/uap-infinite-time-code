#!/usr/bin/env python3
"""Two square panels from the Duffing B-type experiment, for main-text Figure 4:
(A) the sup-error map with the separatrix collar, (B) the delta(eps) curve. Large
fonts (these render small in a 4-panel row) and no in-panel legends (that info is in
the caption). Reuses the cached Duffing Neural ODE. Run in the iam env."""
import os, numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from uap.common import harness as H
from uap.demos2d import run_duffing as RD

OUT = RD.OUT
plt, sns = H.setup_style()
plt.rcParams.update({"font.size": 26, "axes.labelsize": 32, "axes.titlesize": 28,
                     "xtick.labelsize": 26, "ytick.labelsize": 26, "lines.linewidth": 3.0,
                     "axes.linewidth": 1.4})

model = RD.get_model()
(xlo, xhi), (ylo, yhi) = RD.sys.domain
ng = 90
gx = np.linspace(xlo, xhi, ng); gy = np.linspace(ylo, yhi, ng)
XX, YY = np.meshgrid(gx, gy); X0 = np.stack([XX.ravel(), YY.ravel()], -1)
ts, sup, Yt, Yh = H.sup_traj_error(RD.sys, model, X0, T=RD.sys.T_eval, dt=0.01, save_every=5)
agree = (np.sign(Yt[-1][:, 0]) == np.sign(Yh[-1][:, 0]))
D = 2.0
sep_true = RD.trace_separatrix(RD.sys.f_np)
sep_hat = RD.trace_separatrix(H.field_np_from_model(model))

# ---- (A) sup-error map (white = true separatrix, dashed = NODE separatrix) ---
fig, ax = plt.subplots(figsize=(5.0, 4.6))
pc = ax.pcolormesh(XX, YY, sup.reshape(ng, ng), shading="auto", cmap="rocket",
                   norm=mcolors.LogNorm(vmin=1e-3, vmax=D))
ax.plot(sep_true[:, 0], sep_true[:, 1], color="white", lw=3.0)
ax.plot(sep_hat[:, 0], sep_hat[:, 1], color=H.OKABE_ITO[3], lw=2.4, ls="--")
ax.plot([1, -1], [0, 0], "o", color="w", ms=11, mec="k", mew=1.4)
ax.plot([0], [0], "s", color="w", ms=11, mec="k", mew=1.4)
ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=22)
sns.despine(fig=fig); fig.tight_layout()
fig.savefig(os.path.join(OUT, "node_duffing_single.pdf"))
fig.savefig(os.path.join(OUT, "node_duffing_single.png"))

# ---- (B) delta(eps) curve (dotted line = basin-flip floor) -------------------
eps_grid = np.logspace(-4, np.log10(3), 80)
delta = H.eps_delta_curve(sup, eps_grid)
fig, ax = plt.subplots(figsize=(5.0, 4.6))
ax.semilogx(eps_grid, delta, color=H.OKABE_ITO[0], lw=3.4)
ax.axhline(1 - agree.mean(), color=H.OKABE_ITO[1], ls=":", lw=2.6)
ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel(r"$\delta(\varepsilon)$")
ax.set_xlim(1e-4, 3); ax.set_ylim(-0.03, 1.03)
ax.set_xticks([1e-4, 1e-2, 1e0]); ax.set_yticks([0, 0.5, 1.0])
sns.despine(fig=fig); fig.tight_layout()
fig.savefig(os.path.join(OUT, "node_duffing_epsdelta.pdf"))
fig.savefig(os.path.join(OUT, "node_duffing_epsdelta.png"))
print(f"saved duffing panels; basin {agree.mean()*100:.2f}% delta(0.3)={(sup>0.3).mean():.4f}", flush=True)
