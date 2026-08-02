#!/usr/bin/env python3
"""
System 2 -- P-type error on a limit cycle (Van der Pol).

Target:  x' = y ,  y' = mu (1 - x^2) y - x        (mu = 1)
One hyperbolic stable limit cycle.  We train a Neural ODE in C^1, then show:
  * the learned cycle has the right SHAPE (small geometric error) but a slightly
    wrong PERIOD,
  * that period mismatch produces PHASE DRIFT: the trajectory error grows toward
    the cycle diameter, so sup_t error is large even though the field error is
    tiny -- P-TYPE error, invisible to a short-horizon loss,
  * the detection horizon scales as 1 / (period error), matching the theory.
"""
from uap import config
import os, numpy as np, torch
from uap.common import harness as H

OUT = config.FIGDIR
MOD = config.MODELDIR
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)

MU = 1.0
sys = H._vanderpol(MU)


def get_model(width=64, depth=2, epochs=5000, retrain=False):
    path = os.path.join(MOD, f"vanderpol_w{width}_d{depth}.pt")
    model = H.MLPField(2, width, depth)
    if os.path.exists(path) and not retrain:
        model.load_state_dict(torch.load(path)); model.eval()
        print(f"  loaded {path}")
        return model
    print(f"  training vanderpol width={width} depth={depth} epochs={epochs}")
    model = H.train_field(sys, width=width, depth=depth, n_samples=8000,
                          c1=1.0, epochs=epochs, lr=2e-3, seed=0)
    torch.save(model.state_dict(), path)
    return model


def measure_period(field, x0=(2.0, 0.0), T=150.0, dt=0.006, n_skip_cycles=5):
    """Poincaré section y=0, x>0, dy/dt<0 crossing (top of the loop).
    Returns mean period from late crossings and one settled cycle sample."""
    _, Y = H.rk4(field, np.array([x0]), T, dt, save_every=1)
    Y = Y[:, 0, :]
    y = Y[:, 1]
    # upward-to-downward crossings of y=0 with x>0
    cross = []
    for k in range(1, len(Y)):
        if Y[k, 0] > 0 and y[k-1] > 0 and y[k] <= 0:
            # linear interp for crossing time
            frac = y[k-1] / (y[k-1] - y[k])
            cross.append((k - 1 + frac) * dt)
    cross = np.array(cross)
    if len(cross) < n_skip_cycles + 3:
        return np.nan, Y
    periods = np.diff(cross[n_skip_cycles:])
    return periods.mean(), Y


def settle_on_cycle(field, x0=(2.0, 0.0), T_settle=60.0, dt=0.002):
    _, Y = H.rk4(field, np.array([x0]), T_settle, dt, save_every=1)
    return Y[-1, 0, :]                              # a point on the cycle


def main():
    model = get_model()
    err = H.field_errors(sys, model, n_grid=120)
    print("  field errors:", {k: round(v, 5) for k, v in err.items()})

    fh = H.field_np_from_model(model)
    T_true, _ = measure_period(sys.f_np)
    T_hat, _ = measure_period(fh)
    eT = abs(T_hat - T_true) / T_true
    print(f"  true period   T  = {T_true:.5f}  (analytic ~6.6633 for mu=1)")
    print(f"  learned period T̂ = {T_hat:.5f}   relative error e_T = {eT:.3e}")

    # ---- phase drift from a common start on the true cycle ------------------
    # detection horizon (phase diff reaches pi) scales as 1/e_T; integrate well
    # past it so the error actually climbs to the cycle diameter.
    tstar_pred = 0.5 / abs(1.0 / T_true - 1.0 / T_hat)
    x0 = settle_on_cycle(sys.f_np)                  # a point on the true cycle
    Tmax = float(min(9000.0, max(400.0, 2.4 * tstar_pred)))
    dt = 0.05
    ts, Yt = H.rk4(sys.f_np, np.array([x0]), Tmax, dt, save_every=4)
    _,  Yh = H.rk4(fh,        np.array([x0]), Tmax, dt, save_every=4)
    d = np.linalg.norm(Yt[:, 0, :] - Yh[:, 0, :], axis=-1)
    diam = 2 * np.max(np.linalg.norm(Yt[:, 0, :], axis=-1))   # cycle diameter
    # the drift envelope first peaks at the full diameter when the phase
    # difference reaches pi, i.e. at t* -- locate that empirically as argmax(d)
    tstar_emp = ts[np.argmax(d)]
    print(f"  cycle diameter D = {diam:.3f}")
    print(f"  detection horizon (envelope peaks at D): predicted t* = "
          f"{tstar_pred:.1f}, empirical argmax = {tstar_emp:.1f}")

    # ---- sup error vs horizon: running max of the phase-drift trajectory ----
    # (reuse the single trajectory already integrated above -- no extra cost)
    cummax = np.maximum.accumulate(d)
    horizons = np.array([5, 20, 80, 300, 1000, 3000, float(ts.max())])
    sup_vs_T = np.array([cummax[min(np.searchsorted(ts, T), len(ts)-1)]
                         for T in horizons])
    for T, s in zip(horizons, sup_vs_T):
        print(f"    horizon T={T:6.0f}  running-sup error {s:.4f}")

    # ---- figure --------------------------------------------------------------
    plt, sns = H.setup_style()
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 24, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "lines.linewidth": 3.0, "axes.linewidth": 1.4})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel A: true vs learned limit cycle -- plot ~2 full periods so the loop closes
    ax = axes[0]
    _, Yt2 = H.rk4(sys.f_np, np.array([x0]), 60, 0.002, save_every=1)
    _, Yh2 = H.rk4(fh,        np.array([x0]), 60, 0.002, save_every=1)
    nn = int(2.05 * T_true / 0.002)
    # true cycle as a wide faded band; learned cycle as a crisp dashed curve on top,
    # so they are clearly two objects even though they coincide in shape
    ax.plot(Yt2[-nn:, 0, 0], Yt2[-nn:, 0, 1], color=H.OKABE_ITO[0], lw=8.0, alpha=0.35,
            solid_capstyle="round", zorder=1)                       # true (band)
    ax.plot(Yh2[-nn:, 0, 0], Yh2[-nn:, 0, 1], color=H.OKABE_ITO[1], lw=2.2,
            ls=(0, (6, 4)), zorder=2)                               # learned (dashed)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.text(0.5, 0.5, r"$e_T = %.1e$" % eT, transform=ax.transAxes, fontsize=19,
            ha="center", va="center")

    # Panel B: running supremum of the trajectory error over the horizon (clean,
    # monotonic) -- rises to the cycle diameter by t*; the 45-cycle window sees ~0
    ax = axes[1]
    cummax = np.maximum.accumulate(d)
    ax.plot(ts, cummax, color=H.OKABE_ITO[0], lw=3.2)
    ax.axhline(diam, color="gray", ls="--", lw=2.2)
    ax.axvline(tstar_pred, color=H.OKABE_ITO[3], ls=":", lw=2.6)
    ax.axvspan(ts[1], 300, color=H.OKABE_ITO[2], alpha=0.22)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$  (horizon)")
    ax.set_ylabel(r"$\sup_{s \leq t}\,\|\varphi - \hat\varphi\|$")
    ax.set_xlim(ts[1], ts.max()); ax.set_ylim(0, 1.08 * diam)

    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "node_vanderpol_ptype.pdf"))
    fig.savefig(os.path.join(OUT, "node_vanderpol_ptype.png"))
    print("  saved figures/node_vanderpol_ptype.{pdf,png}")

    print("\n  SUMMARY (P-type, Van der Pol):")
    print(f"    eta_C0(sup)={err['eta_c0_sup']:.4f}  eta_C1(sup)={err['eta_c1_sup']:.4f}")
    print(f"    period error e_T={eT:.2e}; sup-error grows {sup_vs_T[0]:.3f}"
          f" (T=5) -> {sup_vs_T[-1]:.3f} (T={horizons[-1]:.0f}); "
          f"detection horizon ~1/e_T")


if __name__ == "__main__":
    main()
