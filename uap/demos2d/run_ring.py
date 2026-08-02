#!/usr/bin/env python3
"""
System 3 -- D-type error on a CONTINUOUS ATTRACTOR (ring of equilibria).

Target:  z' = (1 - ||z||^2) z          -> unit circle is a continuum of fixed
points (a ring attractor), normally hyperbolic (normal rate -2), tangentially
neutral.  We train a Neural ODE in C^1 and show:
  * the ring PERSISTS (normal contraction preserved) -- epsilon-delta close,
  * but the continuum is DISCRETIZED: the learned tangential field has finitely
    many zeros, i.e. finitely many isolated fixed points (alternating sinks and
    sources) -- the non-wandering set is never homeomorphic to the circle,
  * trajectories settle onto that discrete set: D-TYPE error, of size ~ the tile
    spacing, while the time-averaged L^2 error is small.
"""
from uap import config
import os, numpy as np, torch
from uap.common import harness as H

OUT = config.FIGDIR
MOD = config.MODELDIR
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)

sys = H._ring()


def get_model(width=64, depth=2, epochs=5000, retrain=False):
    path = os.path.join(MOD, f"ring_w{width}_d{depth}.pt")
    model = H.MLPField(2, width, depth)
    if os.path.exists(path) and not retrain:
        model.load_state_dict(torch.load(path)); model.eval()
        print(f"  loaded {path}")
        return model
    print(f"  training ring width={width} depth={depth} epochs={epochs}")
    model = H.train_field(sys, width=width, depth=depth, n_samples=8000,
                          c1=1.0, epochs=epochs, lr=2e-3, seed=0)
    torch.save(model.state_dict(), path)
    return model


def ring_tangential(field, n=2000):
    """Tangential component of the field along the unit circle, T(theta)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.stack([np.cos(th), np.sin(th)], axis=-1)
    F = field(P)
    tang = np.stack([-np.sin(th), np.cos(th)], axis=-1)
    rad = np.stack([np.cos(th), np.sin(th)], axis=-1)
    Tcomp = np.sum(F * tang, axis=-1)
    Rcomp = np.sum(F * rad, axis=-1)
    return th, Tcomp, Rcomp


def count_fixed_points(th, Tcomp):
    """Zeros of T(theta) with sign change; sink if dT/dtheta<0, else source."""
    sinks, sources = [], []
    n = len(th)
    for k in range(n):
        a, b = Tcomp[k], Tcomp[(k + 1) % n]
        if a == 0 or (a < 0) != (b < 0):
            frac = a / (a - b) if (a - b) != 0 else 0.0
            thc = th[k] + frac * (2 * np.pi / n)
            if b - a < 0:                          # decreasing -> sink
                sinks.append(thc % (2 * np.pi))
            else:
                sources.append(thc % (2 * np.pi))
    return np.array(sinks), np.array(sources)


def main():
    model = get_model()
    err = H.field_errors(sys, model)
    print("  field errors:", {k: round(v, 5) for k, v in err.items()})

    fh = H.field_np_from_model(model)
    th, Tcomp, Rcomp = ring_tangential(fh)
    sinks, sources = count_fixed_points(th, Tcomp)
    print(f"  learned fixed points on the ring: {len(sinks)} sinks + "
          f"{len(sources)} sources = {len(sinks)+len(sources)} total")
    print(f"  max |tangential| along ring = {np.abs(Tcomp).max():.4e} "
          f"(true: 0 exactly)")
    print(f"  radial component along ring: mean {Rcomp.mean():.4e} "
          f"(true 0); normal Jacobian preserved => ring persists")

    # ---- settle many ICs near the ring, see them cluster at sinks -----------
    rng = np.random.default_rng(1)
    th0 = rng.uniform(0, 2 * np.pi, 800)
    r0 = 1 + rng.uniform(-0.15, 0.15, 800)
    X0 = np.stack([r0 * np.cos(th0), r0 * np.sin(th0)], axis=-1)
    _, Y = H.rk4(fh, X0, 80.0, 0.02, save_every=100)
    end = Y[-1]
    th_end = np.arctan2(end[:, 1], end[:, 0]) % (2 * np.pi)

    # along-ring settling error: distance from each start angle to nearest sink
    def nearest_sink_gap(angles, sinks):
        if len(sinks) == 0:
            return np.full_like(angles, np.nan)
        d = np.abs(((angles[:, None] - sinks[None, :] + np.pi) % (2 * np.pi)) - np.pi)
        return d.min(axis=1)
    gaps = nearest_sink_gap(th0, sinks)
    print(f"  along-ring settling: max angular gap to nearest sink "
          f"{np.nanmax(gaps):.3f} rad ≈ pi/{np.pi/np.nanmax(gaps):.1f}")

    # ---- time-averaged L2 error (small despite wrong topology) ---------------
    ts, Yt = H.rk4(sys.f_np, X0, 80.0, 0.02, save_every=25)
    _,  Yh = H.rk4(fh,        X0, 80.0, 0.02, save_every=25)
    d2 = np.linalg.norm(Yt - Yh, axis=-1) ** 2
    tail = ts > 0.5 * ts[-1]
    L2 = np.sqrt(d2[tail].mean())
    print(f"  time-averaged L2 error (tail) = {L2:.4e}")

    # ---- figure --------------------------------------------------------------
    plt, sns = H.setup_style()
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: learned tangential field along the ring, zeros marked
    ax = axes[0]
    ax.plot(th, Tcomp, color=H.OKABE_ITO[0], lw=2.2)
    ax.axhline(0, color="gray", lw=1, ls=":")
    if len(sinks):
        ax.plot(sinks, np.zeros_like(sinks), "v", color=H.OKABE_ITO[2],
                ms=9, label=f"sinks ({len(sinks)})")
    if len(sources):
        ax.plot(sources, np.zeros_like(sources), "^", color=H.OKABE_ITO[1],
                ms=9, label=f"sources ({len(sources)})")
    ax.set_xlabel(r"$\theta$ on the ring")
    ax.set_ylabel(r"learned tangential $\hat f_\theta(\theta)$")
    ax.set_xlim(0, 2 * np.pi)
    ax.legend(loc="upper right", fontsize=12)

    # Panel B: settling-angle histogram (clusters at the discrete sinks)
    ax = axes[1]
    ax.hist(th_end, bins=120, color=H.OKABE_ITO[0], alpha=0.85)
    for s in sinks:
        ax.axvline(s, color=H.OKABE_ITO[2], lw=1.2, alpha=0.7)
    ax.set_xlabel(r"settled angle $\theta$")
    ax.set_ylabel("count")
    ax.set_xlim(0, 2 * np.pi)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    H.panel_labels(axes)
    fig.savefig(os.path.join(OUT, "node_ring_dtype.pdf"))
    fig.savefig(os.path.join(OUT, "node_ring_dtype.png"))
    print("  saved figures/node_ring_dtype.{pdf,png}")

    print("\n  SUMMARY (D-type, ring):")
    print(f"    eta_C0(sup)={err['eta_c0_sup']:.4f}  eta_C1(sup)={err['eta_c1_sup']:.4f}")
    print(f"    {len(sinks)+len(sources)} isolated FPs (never ~ circle); "
          f"time-avg L2 {L2:.2e}; max settling gap {np.nanmax(gaps):.3f} rad")


if __name__ == "__main__":
    main()
