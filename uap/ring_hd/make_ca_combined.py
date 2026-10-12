#!/usr/bin/env python3
"""Combined continuous-attractor (D-type) figure: the toy ring Neural ODE (top) and
the real trained head-direction network (bottom) show the same two things: the
along-ring drift with its alternating sinks (v) and sources (^), and the settling
histogram of held states. Both discretize a continuum into isolated headings, the
D-type error of Theorem 5. Run in the uap-infinite-time environment."""
import os, numpy as np
from uap.common import harness as H
from uap.demos2d import run_ring as RR
from uap.ring_hd import fca_ring_btca as BT
from matplotlib.ticker import MaxNLocator

OUT = RR.OUT
plt, sns = H.setup_style()
plt.rcParams.update({"font.size": 20, "axes.labelsize": 23, "xtick.labelsize": 18,
                     "ytick.labelsize": 18, "lines.linewidth": 2.6, "axes.linewidth": 1.4})
OK = H.OKABE_ITO
XT = [0, np.pi, 2 * np.pi]; XL = ["0", r"$\pi$", r"$2\pi$"]

# ---- toy ring ---------------------------------------------------------------
m = RR.get_model()
fh = H.field_np_from_model(m)
th_r, Tr, _ = RR.ring_tangential(fh)
sink_r, src_r = RR.count_fixed_points(th_r, Tr)
rng = np.random.default_rng(1)
th0 = rng.uniform(0, 2 * np.pi, 800); r0 = 1 + rng.uniform(-0.15, 0.15, 800)
X0 = np.stack([r0 * np.cos(th0), r0 * np.sin(th0)], -1)
_, Y = H.rk4(fh, X0, 80.0, 0.02, save_every=100)
settle_r = np.arctan2(Y[-1][:, 1], Y[-1][:, 0]) % (2 * np.pi)

# ---- real head-direction network -------------------------------------------
net, name = BT.load_btca(0); dt = net["dt"]
th_b, vth = BT.ring_drift(net)
z = np.load(os.path.join(os.path.dirname(BT.__file__), "models", "btca_ring_cache.npz"))
thf, lead, settle_b = z["thf"], z["lead"], z["th_settle"]
sink_b, src_b = thf[lead < 1e-6], thf[lead >= 1e-6]
print(f"toy ring {len(sink_r)} sinks/{len(src_r)} sources | BTCA {len(sink_b)} sinks/{len(src_b)} sources", flush=True)

# ---- 2x2 figure -------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))


def drift_panel(a, x, v, sinks, sources, ylab):
    a.plot(x, v, color=OK[0], lw=2.6)
    a.axhline(0, color="gray", lw=1.2, ls=":")
    a.plot(sinks, np.zeros_like(sinks), "v", color=OK[2], ms=11, mec="white", mew=0.8, zorder=3)
    a.plot(sources, np.zeros_like(sources), "^", color=OK[1], ms=11, mec="white", mew=0.8, zorder=3)
    a.set_xlim(0, 2 * np.pi); a.set_xticks(XT); a.set_xticklabels(XL); a.set_ylabel(ylab)


def hist_panel(a, settled, sinks, ylab):
    a.hist(settled, bins=100, color=OK[0], alpha=0.85)
    for s in sinks:
        a.axvline(s, color=OK[2], lw=0.9, alpha=0.55)
    a.set_xlim(0, 2 * np.pi); a.set_xticks(XT); a.set_xticklabels(XL); a.set_ylabel(ylab)
    a.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))


drift_panel(ax[0, 0], th_r, Tr, sink_r, src_r, r"tangential drift $\hat f_\theta$")
hist_panel(ax[0, 1], settle_r, sink_r, "count")
drift_panel(ax[1, 0], th_b, vth * dt, sink_b, src_b, r"along-ring drift $\dot\theta$")
hist_panel(ax[1, 1], settle_b, sink_b, "count")
ax[1, 0].set_xlabel(r"heading $\theta$"); ax[1, 1].set_xlabel(r"settled heading $\theta$")
ax[0, 0].set_title("toy ring Neural ODE", fontsize=21, pad=10)
ax[0, 1].set_title("toy ring Neural ODE", fontsize=21, pad=10)
ax[1, 0].set_title("trained head-direction network", fontsize=21, pad=10)
ax[1, 1].set_title("trained head-direction network", fontsize=21, pad=10)

sns.despine(fig=fig)
fig.tight_layout(rect=[0, 0, 1, 0.96])
H.panel_labels(ax)
fig.savefig(os.path.join(OUT, "node_ca_combined.pdf"))
fig.savefig(os.path.join(OUT, "node_ca_combined.png"))
print("saved node_ca_combined", flush=True)
