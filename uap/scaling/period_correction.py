#!/usr/bin/env python3
"""
DEMONSTRATION of Theorem 4's period-correction construction, end to end.

Not "matched vs unmatched by fiat" (as in run_vanderpol.py), but the actual
algorithm: train a NODE on a limit-cycle system, MEASURE the learned period,
CONSTRUCT the localized rescaling, and verify the phase drift FREEZES.

Two regimes, exactly as in the paper:
  (A) single cycle (Van der Pol): global scaling  f_hat_corr = (T_hat/T) f_hat
      (the bump is psi == 1).  Exact period matching.
  (B) multiple cycles (two coaxial limit cycles of different periods): LOCALIZED
      bumps psi_i (=1 on cycle i, ~0 on the others), solve the (diagonally
      dominant) linear system for alpha_i, and correct both periods at once.
      This exercises the genuinely novel machinery (Lemmas 15-18).

Run in the uap-infinite-time environment.
"""
from uap import config
import os, numpy as np, torch
from uap.common import harness as H

OUT = config.FIGDIR
MOD = config.MODELDIR
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)


# ----------------------------------------------------------------- period tools
def measure_period(field, x0, T=150.0, dt=0.006, n_skip=6):
    """Poincare section: upward-to-downward y=0 crossings with x>0."""
    _, Y = H.rk4(field, np.array([x0]), T, dt, save_every=1)
    Y = Y[:, 0, :]; y = Y[:, 1]
    cross = []
    for k in range(1, len(Y)):
        if Y[k, 0] > 0 and y[k-1] > 0 and y[k] <= 0:
            frac = y[k-1] / (y[k-1] - y[k])
            cross.append((k - 1 + frac) * dt)
    cross = np.array(cross)
    if len(cross) < n_skip + 3:
        return np.nan
    return float(np.diff(cross[n_skip:]).mean())


def settle(field, x0, T=60.0, dt=0.005):
    _, Y = H.rk4(field, np.array([x0]), T, dt, save_every=1)
    return Y[-1, 0, :]


# ================================================================ (A) 1 cycle
def demo_single_cycle():
    print("=" * 70); print("(A) SINGLE CYCLE  (Van der Pol): global scaling c* = T_hat/T")
    print("=" * 70)
    sys = H._vanderpol(1.0)
    model = H.MLPField(2, 64, 2)
    path = os.path.join(MOD, "vanderpol_w64_d2.pt")
    model.load_state_dict(torch.load(path)); model.eval()
    fh = H.field_np_from_model(model)                      # base learned field

    x0 = settle(sys.f_np, [2.0, 0.0])
    T_true = measure_period(sys.f_np, [2.0, 0.0])
    T_hat = measure_period(fh, [2.0, 0.0])
    cstar = T_hat / T_true                                  # correction factor
    fh_corr = (lambda base, c: (lambda P: c * base(P)))(fh, cstar)
    T_corr = measure_period(fh_corr, [2.0, 0.0])
    print(f"  true period      T   = {T_true:.5f}")
    print(f"  learned period   T_hat = {T_hat:.5f}   (e_T = {abs(T_hat-T_true)/T_true:.2e})")
    print(f"  scaling          c*  = T_hat/T = {cstar:.6f}")
    print(f"  corrected period T_corr = {T_corr:.5f}   "
          f"(e_T = {abs(T_corr-T_true)/T_true:.2e})")

    # phase drift: true vs base vs corrected from a common cycle point
    e_base = abs(T_hat - T_true) / T_true
    tstar = 0.5 / abs(1.0/T_true - 1.0/T_hat)
    Tmax = float(min(9000.0, 2.4 * tstar)); dt = 0.05
    ts, Yt = H.rk4(sys.f_np, np.array([x0]), Tmax, dt, save_every=4)
    _,  Yb = H.rk4(fh,        np.array([x0]), Tmax, dt, save_every=4)
    _,  Yc = H.rk4(fh_corr,   np.array([x0]), Tmax, dt, save_every=4)
    d_base = np.linalg.norm(Yt[:, 0] - Yb[:, 0], axis=-1)
    d_corr = np.linalg.norm(Yt[:, 0] - Yc[:, 0], axis=-1)
    diam = 2 * np.max(np.linalg.norm(Yt[:, 0], axis=-1))
    print(f"  cycle diameter D = {diam:.3f}")
    print(f"  sup error to t={Tmax:.0f}:  base {d_base.max():.3f} (~D, drifts) ; "
          f"corrected {d_corr.max():.3f} (frozen)")
    return dict(sys=sys, ts=ts, d_base=d_base, d_corr=d_corr, diam=diam,
                tstar=tstar, T_true=T_true, T_hat=T_hat, T_corr=T_corr,
                Yt=Yt, Yb=Yb, Yc=Yc)


# ================================================================ (B) 2 cycles
def two_cycle_system(w1=1.0, w2=2.0):
    """Radial: stable cycles at r=1 (period 2pi/w1) and r=3 (period 2pi/w2),
    unstable cycle at r=2.  rdot = -(r-1)(r-2)(r-3),  thetadot = omega(r)."""
    def omega(r):
        return w1 + (w2 - w1) * (r - 1.0) / 2.0        # w1 at r=1, w2 at r=3
    def f(z):
        x, y = z[..., 0], z[..., 1]
        r = np.hypot(x, y) + 1e-12
        rdot = -(r-1)*(r-2)*(r-3)
        thdot = omega(r)
        ux, uy = x/r, y/r
        return np.stack([rdot*ux - r*thdot*uy, rdot*uy + r*thdot*ux], axis=-1)
    def ft(z):
        x, y = z[..., 0], z[..., 1]
        r = torch.hypot(x, y) + 1e-12
        rdot = -(r-1)*(r-2)*(r-3)
        thdot = w1 + (w2 - w1)*(r-1.0)/2.0
        ux, uy = x/r, y/r
        return torch.stack([rdot*ux - r*thdot*uy, rdot*uy + r*thdot*ux], dim=-1)
    def jac(z):                                        # numeric (finite diff)
        N = z.shape[0]; J = np.zeros((N, 2, 2)); h = 1e-5
        for k in range(2):
            dz = np.zeros_like(z); dz[:, k] = h
            J[:, :, k] = (f(z+dz) - f(z-dz)) / (2*h)
        return J
    T1, T2 = 2*np.pi/w1, 2*np.pi/w2
    return H.System("twocyc", f, jac, ft, ((-3.5, 3.5), (-3.5, 3.5)), 60.0,
                    r"two cycles"), T1, T2


def measure_period_radial(field, r0, T=140.0, dt=0.005, n_skip=6):
    """Period of the cycle reached from radius r0: crossings of the +x axis."""
    x0 = [r0, 0.0]
    _, Y = H.rk4(field, np.array([x0]), T, dt, save_every=1)
    Y = Y[:, 0, :]; y = Y[:, 1]
    cross = [ (k-1 + y[k-1]/(y[k-1]-y[k]))*dt for k in range(1, len(Y))
              if Y[k,0] > 0 and y[k-1] > 0 and y[k] <= 0 ]
    cross = np.array(cross)
    if len(cross) < n_skip+3: return np.nan
    return float(np.diff(cross[n_skip:]).mean())


def demo_two_cycles(retrain=False):
    print("\n" + "=" * 70)
    print("(B) TWO CYCLES: localized bumps, simultaneous period correction")
    print("=" * 70)
    sys2, T1, T2 = two_cycle_system(1.0, 2.0)
    print(f"  target periods:  T1 (r=1) = {T1:.5f} ,  T2 (r=3) = {T2:.5f}")
    path = os.path.join(MOD, "twocyc_w96_d2.pt")
    model = H.MLPField(2, 96, 2)
    if os.path.exists(path) and not retrain:
        model.load_state_dict(torch.load(path)); model.eval()
        print("  loaded trained NODE")
    else:
        print("  training NODE on the two-cycle field ...")
        model = H.train_field(sys2, width=96, depth=2, n_samples=9000, c1=0.5,
                              epochs=4000, lr=2e-3, seed=0, verbose=False,
                              batch_size=1024)
        torch.save(model.state_dict(), path)
    fh = H.field_np_from_model(model)

    That1 = measure_period_radial(fh, 1.0)
    That2 = measure_period_radial(fh, 3.0)
    print(f"  learned periods: T1_hat = {That1:.5f} (e {abs(That1-T1)/T1:.2e}) ,"
          f"  T2_hat = {That2:.5f} (e {abs(That2-T2)/T2:.2e})")

    # localized radial bumps: psi_i = 1 near r=r_i, decaying with width wb
    wb = 0.5
    def psi(P, rc):
        r = np.hypot(P[..., 0], P[..., 1])
        return np.exp(-0.5 * ((r - rc) / wb) ** 2)
    # leakage of each bump onto the other cycle (diagonal-dominance check)
    leak12 = float(np.exp(-0.5 * ((3.0 - 1.0) / wb) ** 2))
    print(f"  bump leakage psi_1 at r=3 (and psi_2 at r=1): {leak12:.2e}  "
          f"(=> Jacobian strongly diagonally dominant)")

    # corrected field f_hat = (1 + a1 psi_1 + a2 psi_2) f_hat_base ; solve for a
    def corrected(a1, a2):
        return lambda P: (1 + a1*psi(P, 1.0) + a2*psi(P, 3.0))[..., None] * fh(P)
    # diagonal guess alpha_i = (That_i - T_i)/T_i, then 1 Newton-ish refinement
    a = np.array([(That1 - T1)/T1, (That2 - T2)/T2])
    for it in range(4):
        fc = corrected(a[0], a[1])
        p1 = measure_period_radial(fc, 1.0); p2 = measure_period_radial(fc, 3.0)
        r1, r2 = (p1 - T1)/T1, (p2 - T2)/T2            # relative period residual
        if max(abs(r1), abs(r2)) < 1e-4: break
        # each cycle's period ~ base/(1+a_i); da_i ~ (1+a_i)*residual_i
        a[0] += (1 + a[0]) * r1
        a[1] += (1 + a[1]) * r2
    fc = corrected(a[0], a[1])
    p1c = measure_period_radial(fc, 1.0); p2c = measure_period_radial(fc, 3.0)
    print(f"  solved corrections: alpha = ({a[0]:+.4f}, {a[1]:+.4f})")
    print(f"  corrected periods:  T1 = {p1c:.5f} (e {abs(p1c-T1)/T1:.2e}) ,"
          f"  T2 = {p2c:.5f} (e {abs(p2c-T2)/T2:.2e})")
    return dict(sys2=sys2, fh=fh, fc=fc, T1=T1, T2=T2, That1=That1, That2=That2,
                p1c=p1c, p2c=p2c, alpha=a, psi=psi)


def main():
    A = demo_single_cycle()
    B = demo_two_cycles()

    # -------- figure --------
    plt, sns = H.setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: single-cycle phase drift, uncorrected vs corrected
    ax0 = ax[0]
    ax0.plot(A["ts"], A["d_base"], color=H.OKABE_ITO[1], lw=1.4,
             label=r"uncorrected ($T_{\hat f}\neq T$): drifts to $D$")
    ax0.plot(A["ts"], A["d_corr"], color=H.OKABE_ITO[2], lw=2.2,
             label=r"corrected ($c^*=\hat T/T$): frozen")
    ax0.axhline(A["diam"], color="gray", ls="--", lw=1.3, label=r"diameter $D$")
    ax0.set_xlabel(r"$t$"); ax0.set_ylabel(r"$\|\varphi-\hat\varphi\|$")
    ax0.set_xlim(0, A["ts"].max())
    ax0.legend(fontsize=11, loc="center right")
    ax0.set_title(r"(A) single cycle: drift frozen by $c^*=\hat T/T$")

    # Panel B: two-cycle periods, target vs learned vs corrected
    ax1 = ax[1]
    labels = [r"cycle 1 ($r{=}1$)", r"cycle 2 ($r{=}3$)"]
    xpos = np.array([0, 1]); wbar = 0.25
    tgt = [B["T1"], B["T2"]]; lrn = [B["That1"], B["That2"]]; cor = [B["p1c"], B["p2c"]]
    ax1.bar(xpos - wbar, tgt, wbar, color="k", label="target $T_i$")
    ax1.bar(xpos, lrn, wbar, color=H.OKABE_ITO[1], label=r"learned $\hat T_i$")
    ax1.bar(xpos + wbar, cor, wbar, color=H.OKABE_ITO[2], label="corrected")
    ax1.set_xticks(xpos); ax1.set_xticklabels(labels)
    ax1.set_ylabel("period")
    ax1.legend(fontsize=11)
    ax1.set_title(r"(B) two cycles: localized bumps match both periods")

    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "node_period_correction.pdf"))
    fig.savefig(os.path.join(OUT, "node_period_correction.png"))
    print("\n  saved figures/node_period_correction.{pdf,png}")


if __name__ == "__main__":
    main()
