#!/usr/bin/env python3
"""
F_CA (D-type) on a REAL trained RNN: the BTCA recttanh S^1 ring attractor.

Checkpoints from the "Back to the Continuous Attractor" (BTCA) project
(Sagodi et al.; recttanh_stability checkpoints): 64-unit recttanh RNNs trained to
integrate a scalar angular velocity on the circle. Point BTCA_CHECKPOINTS_DIR (in
.env) at the directory of result_dict_*.pickle files; see the README.

  h_{t+1} = (1-dt) h + dt * rect_tanh(h @ W_rec.T + u @ W_i + b_rec),  rect_tanh=max(tanh,0)
  autonomous field (u=0):  f(h) = -h + rect_tanh(h @ W_rec.T + b_rec)
  readout: out = h @ W_o ;  theta = atan2(out_y, out_x)
  cue->hidden ring map:  h(theta) = rect_tanh([cos theta, sin theta] @ oth)

Being a FINITE network it cannot hold a true continuum: its autonomous ring is
discretized into finitely many isolated fixed points (alternating sinks/sources)
with a small residual along-ring drift -- the D-type discretization of Theorem 5,
in a real trained model (and the mechanism of head-direction drift). We trace the
ring, compute the autonomous on-ring dtheta/dt, count its zeros (the tiles), and
show trajectories settle onto the discrete sinks.  Run in the uap-infinite-time environment.
"""
import os, glob, pickle
import numpy as np
from uap.common import harness as H

from uap import config
BTCA = config.BTCA_CHECKPOINTS_DIR   # set BTCA_CHECKPOINTS_DIR in .env (see README)
OUT = config.FIGDIR


def rect_tanh(x): return np.maximum(np.tanh(x), 0.0)

def load_btca(idx=0):
    paths = sorted(glob.glob(os.path.join(BTCA, "result_dict_*.pickle")))
    with open(paths[idx], "rb") as f:
        r = pickle.load(f)
    wi, wrec, wo, brec, h0, oth = [np.asarray(a, float) for a in r["weights_last"]]
    return dict(wi=wi, wrec=wrec, wo=wo, brec=brec, oth=oth,
                dt=r["training_kwargs"]["dt_rnn"]), os.path.basename(paths[idx])

def forward(inp, h0, net):
    B, T, _ = inp.shape; h = np.array(h0, float); dt = net["dt"]
    S = np.zeros((B, T, net["wrec"].shape[0]))
    for t in range(T):
        rec = rect_tanh(h @ net["wrec"].T + inp[:, t] @ net["wi"] + net["brec"])
        h = (1-dt)*h + dt*rec; S[:, t] = h
    return S

def decode(S, net):
    o = S @ net["wo"]; return np.arctan2(o[..., 1], o[..., 0])

def vf(Hh, net):                                   # autonomous dh/dt
    return -Hh + rect_tanh(Hh @ net["wrec"].T + net["brec"])

def jac(h, net):                                   # Jacobian of vf at one point
    Wr, br = net["wrec"], net["brec"]
    z = h @ Wr.T + br
    d = np.where(np.tanh(z) > 0, 1 - np.tanh(z)**2, 0.0)   # rect_tanh'(z)
    return -np.eye(len(h)) + (d[:, None] * Wr)

def newton_fp(h0, net, iters=60):
    h = h0.copy()
    for _ in range(iters):
        fh = vf(h[None], net)[0]
        if np.linalg.norm(fh) < 1e-11: break
        try: h = h - np.linalg.solve(jac(h, net), fh)
        except np.linalg.LinAlgError: break
    return h, np.linalg.norm(vf(h[None], net)[0])

def find_fixed_points(net, n_seed=400, relax=200):
    """True FPs of the 64-D autonomous field: relax cue points, Newton-polish,
    dedupe in state space, classify by the leading Jacobian eigenvalue."""
    dt = net["dt"]
    th0 = np.linspace(0, 2*np.pi, n_seed, endpoint=False)
    s = rect_tanh(np.stack([np.cos(th0), np.sin(th0)], -1) @ net["oth"])
    for _ in range(relax):
        s = (1-dt)*s + dt*rect_tanh(s @ net["wrec"].T + net["brec"])
    uniq = []
    for seed in s:
        h, r = newton_fp(seed, net)
        if r < 1e-8 and not any(np.linalg.norm(h-u) < 1e-3 for u in uniq):
            uniq.append(h)
    uniq = np.array(uniq)
    th = decode(uniq, net) % (2*np.pi)
    lead = np.array([np.linalg.eigvals(jac(h, net)).real.max() for h in uniq])
    o = np.argsort(th)
    return uniq[o], th[o], lead[o]        # states, headings, leading Re(eig)


CKPT_IDX = 7


def ring_drift(net, n=4000, relax=15, win=15):
    """Residual along-ring drift dtheta/dt(theta): relax cue points onto the slow
    manifold, order by heading, take a finite-difference tangent (NOT a smooth
    Fourier fit, which would round over the near-marginal tiles) and project the
    autonomous field onto it. Lightly smoothed to tame FD noise."""
    dt = net["dt"]
    th0 = np.linspace(0, 2*np.pi, n, endpoint=False)
    h = rect_tanh(np.stack([np.cos(th0), np.sin(th0)], -1) @ net["oth"])
    for _ in range(relax):
        h = (1-dt)*h + dt*rect_tanh(h @ net["wrec"].T + net["brec"])
    th = decode(h, net) % (2*np.pi); o = np.argsort(th); th, h = th[o], h[o]
    dth = (np.roll(th, -1) - np.roll(th, 1) + np.pi) % (2*np.pi) - np.pi
    tang = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / dth[:, None]
    v = np.sum(vf(h, net)*tang, 1) / np.sum(tang**2, 1)
    k = np.ones(win)/win                                  # circular moving average
    v = np.convolve(np.r_[v[-win:], v, v[:win]], k, "same")[win:-win]
    return th, v


def main():
    net, name = load_btca(CKPT_IDX)
    Hd, dt = net["wrec"].shape[0], net["dt"]
    print(f"loaded BTCA net {name}: N_rec={Hd}, dt={dt}, rect_tanh")

    cache = os.path.join(config.MODELDIR, "btca_ring_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        thf, lead, vth, th_settle = z["thf"], z["lead"], z["vth"], z["th_settle"]
        print("  loaded btca_ring_cache")
    else:
        # true fixed points of the full 64-D autonomous field (Newton)
        _, thf, lead = find_fixed_points(net)
        # residual along-ring drift (finite-difference tangent)
        _, vth = ring_drift(net)
        # settling: held-delay (u=0) from many starts -> discrete headings
        thg = np.linspace(0, 2*np.pi, 1600, endpoint=False)
        hs = rect_tanh(np.stack([np.cos(thg), np.sin(thg)], -1) @ net["oth"])
        for _ in range(30000):
            hs = (1-dt)*hs + dt*rect_tanh(hs @ net["wrec"].T + net["brec"])
        th_settle = decode(hs, net) % (2*np.pi)
        np.savez(cache, thf=thf, lead=lead, vth=vth, th_settle=th_settle)

    sinks, sources = thf[lead < 1e-6], thf[lead >= 1e-6]
    sink_lead = lead[lead < 1e-6]
    print(f"fixed points (Newton, |f|<1e-8): {len(sinks)} sinks + {len(sources)} "
          f"sources = {len(thf)} -> the continuum is tiled by discrete headings")
    print(f"leading eigenvalue of the sinks: median {np.median(sink_lead):+.3e}, "
          f"|max| {np.abs(sink_lead).max():.3e}  (small => weak along-ring restoring, "
          f"close to the continuum limit)")
    print(f"peak residual drift |dtheta/dt| = {np.abs(vth).max()*dt:.3e} rad/step "
          f"(a true continuum has 0; here the heading creeps toward the nearest sink)")

    # ---- figure --------------------------------------------------------------
    plt, sns = H.setup_style()
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.7))
    # (A) the heading ring, tiled by 31 alternating fixed points
    a = ax[0]
    ang = np.linspace(0, 2*np.pi, 400)
    a.plot(np.cos(ang), np.sin(ang), color="0.78", lw=6, solid_capstyle="round",
           zorder=1)                                        # the continuous ring
    a.plot(np.cos(sources), np.sin(sources), "X", color=H.OKABE_ITO[1], ms=10,
           mec="white", mew=1.0, label=f"sources ({len(sources)})", zorder=3)
    a.plot(np.cos(sinks), np.sin(sinks), "o", color=H.OKABE_ITO[2], ms=11,
           mec="white", mew=1.2, label=f"sinks ({len(sinks)})", zorder=4)
    a.set_aspect("equal"); a.axis("off"); a.set_xlim(-1.35, 1.35); a.set_ylim(-1.32, 1.5)
    h_, l_ = a.get_legend_handles_labels()
    a.legend(h_[::-1], l_[::-1], fontsize=12, loc="upper center", ncol=2,
             bbox_to_anchor=(0.5, 1.13), columnspacing=1.0, handletextpad=0.2)
    a.text(0, -1.30, r"peak residual drift $|\dot\theta|\!\approx\!%.1f\times10^{-3}$/step"
           % (np.abs(vth).max()*dt*1e3), ha="center", va="top", fontsize=12.5)
    # (B) autonomous rollouts collapse onto the discrete sinks
    a = ax[1]
    n, _, _ = a.hist(th_settle, bins=120, color=H.OKABE_ITO[0], alpha=0.85)
    ytop = n.max()*1.18
    a.plot(sinks, np.full_like(sinks, n.max()*1.05), "v", color=H.OKABE_ITO[2],
           ms=8, label=f"sinks ({len(sinks)})")
    a.set_ylim(0, ytop)
    a.set_xlabel(r"settled heading $\theta$"); a.set_xlim(0, 2*np.pi)
    a.set_ylabel("count"); a.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    a.legend(fontsize=12, loc="upper right")
    a.set_title(r"continuum $\to$ discrete headings (D-type)", pad=14)
    sns.despine(ax=a); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "node_btca_ring.pdf"))
    fig.savefig(os.path.join(OUT, "node_btca_ring.png"))
    print("  saved figures/node_btca_ring.{pdf,png}")


if __name__ == "__main__":
    main()
