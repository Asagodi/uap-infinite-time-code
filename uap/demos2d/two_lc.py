#!/usr/bin/env python3
"""
Multi-cycle period correction: the localized-bump machinery of Theorem 4.

Target: two coaxial stable limit cycles at r=1 and r=2 (unstable cycle at r=1.5),
with DIFFERENT angular speeds -> different periods T1=2pi, T2=2pi/1.6. A single
global scaling cannot fix both; localized bumps psi_1, psi_2 (=1 near one cycle,
~0 near the other) correct each period independently.

Pipeline:
  train NODE on the field  ->  measure learned periods T1_hat, T2_hat
  ->  build radial bumps psi_i, corrected f_hat = (1 + a1 psi_1 + a2 psi_2) f_hat
  ->  solve (a1, a2) so BOTH periods match (Newton on measured periods; the
      numerical stand-in for the adjoint period-Jacobian inversion, which is
      diagonally dominant because bump leakage is tiny).
"""
from uap import config
import os, numpy as np, torch
from uap.common import harness as H

MOD = config.MODELDIR
OUT = config.FIGDIR


def two_lc_system(r1=1.0, ru=1.5, r2=2.0, k=2.0, w1=1.0, w2=1.6, eps=0.25):
    """Two coaxial stable cycles at r=1, r=2 (unstable at r=1.5), rotation W(r).
    The radial term uses re=sqrt(r^2+eps^2) instead of r, regularizing the 1/r
    coordinate singularity at the origin (bounded, continuous field there) while
    leaving the dynamics for r>>eps unchanged and gentle/fittable."""
    def R(r): return -k*(r-r1)*(r-ru)*(r-r2)
    def W(r): return w1 + (w2-w1)*(r-r1)/(r2-r1)
    def f(z):
        x, y = z[..., 0], z[..., 1]; r = np.hypot(x, y)
        re = np.sqrt(r*r + eps*eps); Rr, Wr = R(r), W(r)
        return np.stack([Rr*x/re - Wr*y, Rr*y/re + Wr*x], axis=-1)
    def ft(z):
        x, y = z[..., 0], z[..., 1]; r = torch.hypot(x, y)
        re = torch.sqrt(r*r + eps*eps)
        Rr = -k*(r-r1)*(r-ru)*(r-r2); Wr = w1 + (w2-w1)*(r-r1)/(r2-r1)
        return torch.stack([Rr*x/re - Wr*y, Rr*y/re + Wr*x], dim=-1)
    def jac(z):
        N = z.shape[0]; J = np.zeros((N, 2, 2)); h = 1e-5
        for kk in range(2):
            dz = np.zeros_like(z); dz[:, kk] = h
            J[:, :, kk] = (f(z+dz) - f(z-dz)) / (2*h)
        return J
    return (H.System("twolc", f, jac, ft, ((-2.6, 2.6), (-2.6, 2.6)), 60.0, "2LC"),
            2*np.pi/w1, 2*np.pi/w2)


def _hroot(y0, y1, m0, m1):
    s = y0/(y0-y1)
    for _ in range(6):
        h00=2*s**3-3*s**2+1; h10=s**3-2*s**2+s; h01=-2*s**3+3*s**2; h11=s**3-s**2
        Hs = h00*y0+h10*m0+h01*y1+h11*m1
        dHs=(6*s**2-6*s)*y0+(3*s**2-4*s+1)*m0+(-6*s**2+6*s)*y1+(3*s**2-2*s)*m1
        s -= Hs/dHs
    return s


def period_from(field, r0, dt=3e-3, T=90.0, n_skip=5):
    """Period of the cycle reached from radius r0 (downward +x-axis crossings)."""
    _, Y = H.rk4(field, np.array([[r0, 0.0]]), T, dt, save_every=1); Y = Y[:, 0]
    y = Y[:, 1]; cr = []
    for k in range(1, len(Y)):
        if Y[k, 0] > 0 and y[k-1] < 0 and y[k] >= 0:      # upward (counterclockwise)
            m0 = field(Y[k-1][None])[0, 1]*dt; m1 = field(Y[k][None])[0, 1]*dt
            cr.append((k-1+_hroot(y[k-1], y[k], m0, m1))*dt)
    cr = np.array(cr)
    if len(cr) < n_skip+2: return np.nan
    d = np.diff(cr[n_skip:])
    return float(d.mean()) if d.std()/max(abs(d.mean()), 1e-9) < 0.02 else np.nan


def main():
    sys2, T1, T2 = two_lc_system()
    print(f"target periods: T1(r=1)={T1:.6f}  T2(r=2)={T2:.6f}")
    path = os.path.join(MOD, "twolc_w128.pt")
    model = H.MLPField(2, 128, 2)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path)); model.eval(); print("loaded NODE")
    else:
        print("training NODE (width 128) ...")
        model = H.train_field(sys2, width=128, depth=2, n_samples=12000, c1=0.4,
                              epochs=6000, lr=2e-3, seed=0, verbose=True,
                              batch_size=2048)
        torch.save(model.state_dict(), path)
    fh = H.field_np_from_model(model)
    err = H.field_errors(sys2, model, n_grid=140)
    print(f"field error eta_C0(sup)={err['eta_c0_sup']:.4f} mean={err['eta_c0_mean']:.4f}")

    Th1 = period_from(fh, 1.0); Th2 = period_from(fh, 2.0)
    print(f"learned periods: T1_hat={Th1}  T2_hat={Th2}")
    if np.isnan(Th1) or np.isnan(Th2):
        print("  -> a cycle did not form; needs more training/width.")
        # diagnostic: where do r=1, r=2 trajectories settle?
        for r0 in (1.0, 2.0):
            _, Y = H.rk4(fh, np.array([[r0, 0.0]]), 40, 0.01, save_every=100)
            rr = np.hypot(Y[-1, 0, 0], Y[-1, 0, 1])
            print(f"     from r={r0}: settled radius ~ {rr:.3f}")
        return

    print(f"  e_T uncorrected: cycle1 {abs(Th1-T1)/T1:.2e}, cycle2 {abs(Th2-T2)/T2:.2e}")

    # localized radial bumps
    wb = 0.35
    psi = lambda P, rc: np.exp(-0.5*((np.hypot(P[..., 0], P[..., 1]) - rc)/wb)**2)
    leak = float(np.exp(-0.5*((2.0-1.0)/wb)**2))
    print(f"  bump leakage psi_i(other cycle) = {leak:.2e}  "
          f"(=> period Jacobian strongly diagonally dominant)")

    def corrected(a):
        return lambda P: (1 + a[0]*psi(P, 1.0) + a[1]*psi(P, 2.0))[..., None]*fh(P)

    # Newton on measured periods (numerical adjoint-Jacobian inversion)
    a = np.array([(Th1-T1)/T1, (Th2-T2)/T2])
    print("  solving (alpha1, alpha2):")
    for it in range(8):
        fc = corrected(a)
        p1 = period_from(fc, 1.0); p2 = period_from(fc, 2.0)
        r1, r2 = (p1-T1)/T1, (p2-T2)/T2
        print(f"    it {it}: a=({a[0]:+.5f},{a[1]:+.5f})  "
              f"periods=({p1:.6f},{p2:.6f})  resid=({r1:+.1e},{r2:+.1e})")
        if max(abs(r1), abs(r2)) < 1e-6:
            break
        a[0] += (1+a[0])*r1; a[1] += (1+a[1])*r2
    fc = corrected(a)
    p1c, p2c = period_from(fc, 1.0), period_from(fc, 2.0)
    print(f"\n  CORRECTED: T1={p1c:.8f} (e {abs(p1c-T1)/T1:.2e}), "
          f"T2={p2c:.8f} (e {abs(p2c-T2)/T2:.2e})")
    print(f"  alpha* = ({a[0]:+.5f}, {a[1]:+.5f})")

    # figure: periods target/learned/corrected
    plt, sns = H.setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    # Panel A: phase portrait (two cycles) + bump profiles
    th = np.linspace(0, 2*np.pi, 400)
    for rc, col in [(1.0, H.OKABE_ITO[0]), (2.0, H.OKABE_ITO[2])]:
        ax[0].plot(rc*np.cos(th), rc*np.sin(th), color=col, lw=1.0, ls=":")
    # learned settled cycles
    for r0, col in [(1.0, H.OKABE_ITO[0]), (2.0, H.OKABE_ITO[2])]:
        _, Y = H.rk4(fh, np.array([[r0, 0.0]]), 40, 0.003, save_every=1)
        ax[0].plot(Y[-3000:, 0, 0], Y[-3000:, 0, 1], color=col, lw=2.0)
    ax[0].set_aspect("equal"); ax[0].set_xlabel(r"$x$"); ax[0].set_ylabel(r"$y$")
    ax[0].set_title("two learned limit cycles")
    # Panel B: periods bars
    xp = np.array([0, 1]); wbar = 0.25
    ax[1].bar(xp-wbar, [T1, T2], wbar, color="k", label="target")
    ax[1].bar(xp, [Th1, Th2], wbar, color=H.OKABE_ITO[1], label=r"learned")
    ax[1].bar(xp+wbar, [p1c, p2c], wbar, color=H.OKABE_ITO[2], label="corrected")
    ax[1].set_xticks(xp); ax[1].set_xticklabels([r"cycle 1", r"cycle 2"])
    ax[1].set_ylabel("period"); ax[1].legend(fontsize=11)
    ax[1].set_title("localized bumps match both periods")
    sns.despine(fig=fig); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "node_two_lc.pdf"))
    fig.savefig(os.path.join(OUT, "node_two_lc.png"))
    print("  saved figures/node_two_lc.{pdf,png}")


if __name__ == "__main__":
    main()
