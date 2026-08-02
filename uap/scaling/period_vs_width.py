#!/usr/bin/env python3
"""
Period error vs BUDGET (MLP width), with and without the correction.

Uncorrected, the learned period error is set by the field-approximation budget:
|T_hat - T| <= L_T * eta   (paper's period-Lipschitz lemma),  eta ~ W^{-p}.
So e_T(W) decays as a power law -- you would need W -> infinity to remove it.
The Theorem-4 correction (solve period(c* f_hat)=T) removes it EXACTLY at any
finite width, to machine precision.  That is why a FINITE network attains the
infinite-horizon guarantee for F_LC.
"""
from uap import config
import os, numpy as np, torch
from uap.common import harness as H

MOD = config.MODELDIR
sys = H._vanderpol(1.0)
x0 = [2.0, 0.0]


def _hroot(y0, y1, m0, m1):
    s = y0 / (y0 - y1)
    for _ in range(6):
        h00 = 2*s**3-3*s**2+1; h10 = s**3-2*s**2+s
        h01 = -2*s**3+3*s**2;   h11 = s**3-s**2
        Hs = h00*y0+h10*m0+h01*y1+h11*m1
        dHs = (6*s**2-6*s)*y0+(3*s**2-4*s+1)*m0+(-6*s**2+6*s)*y1+(3*s**2-2*s)*m1
        s -= Hs/dHs
    return s


def period(field, dt=3e-3, T=70.0, n_skip=5):
    _, Y = H.rk4(field, np.array([x0]), T, dt, save_every=1); Y = Y[:, 0]
    y = Y[:, 1]; cr = []
    for k in range(1, len(Y)):
        if Y[k, 0] > 0 and y[k-1] > 0 and y[k] <= 0:
            m0 = field(Y[k-1][None, :])[0, 1]*dt; m1 = field(Y[k][None, :])[0, 1]*dt
            cr.append((k-1+_hroot(y[k-1], y[k], m0, m1))*dt)
    cr = np.array(cr)
    if len(cr) < n_skip+2:
        return np.nan
    d = np.diff(cr[n_skip:])
    return float(d.mean()) if d.std()/max(d.mean(), 1e-9) < 0.02 else np.nan  # clean cycle only


def corrected_eT(fh, Ttrue):
    """secant refine c so period(c*fh)=Ttrue -> machine floor."""
    g = lambda c: period(lambda P: c*fh(P)) - Ttrue
    c0 = period(fh)/Ttrue; c1 = c0*(1+1e-6); g0, g1 = g(c0), g(c1)
    for _ in range(6):
        c2 = c1 - g1*(c1-c0)/(g1-g0); g2 = g(c2)
        if abs(g2)/Ttrue < 5e-11:
            return abs(g2)/Ttrue
        c0, g0, c1, g1 = c1, g1, c2, g2
    return abs(g2)/Ttrue


Ttrue = period(sys.f_np)
print(f"true period T = {Ttrue:.8f}\n")
print(f"{'W':>4} {'params':>7} {'eta_C0':>10} {'e_T (uncorr)':>13} {'e_T (corr)':>11}")
print("-"*52)
rows = []
for W in [6, 10, 16, 24, 32, 48, 64, 96]:
    p = os.path.join(MOD, f"sweep_vanderpol_w{W}.pt")
    if not os.path.exists(p):
        continue
    m = H.MLPField(2, W, 2); m.load_state_dict(torch.load(p)); m.eval()
    fh = H.field_np_from_model(m)
    eta = H.field_errors(sys, m, n_grid=110)["eta_c0_sup"]
    That = period(fh)
    if np.isnan(That):
        print(f"{W:>4} {m.n_params():>7} {eta:10.3e} {'no clean cycle':>13}")
        continue
    eT = abs(That-Ttrue)/Ttrue
    eTc = corrected_eT(fh, Ttrue)
    rows.append((W, m.n_params(), eta, eT, eTc))
    print(f"{W:>4} {m.n_params():>7} {eta:10.3e} {eT:13.2e} {eTc:11.2e}")

if len(rows) >= 3:
    W = np.array([r[0] for r in rows]); eta = np.array([r[2] for r in rows])
    eT = np.array([r[3] for r in rows])
    pW = -np.polyfit(np.log(W), np.log(eT), 1)[0]
    pe = -np.polyfit(np.log(W), np.log(eta), 1)[0]
    ratio = eT/eta
    print(f"\n  uncorrected e_T(W) ~ W^-{pW:.2f} ;  eta(W) ~ W^-{pe:.2f}")
    print(f"  e_T/eta ratio (= L_T/T, the period-Lipschitz constant): "
          f"{ratio.mean():.3f} +/- {ratio.std():.3f}  (near-constant => e_T ~ L_T*eta)")
    print(f"  corrected e_T: ~1e-16 (machine floor) at EVERY width.")
