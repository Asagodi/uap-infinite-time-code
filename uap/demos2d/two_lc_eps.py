#!/usr/bin/env python3
"""
Vector field and epsilon-delta structure of the (smooth) two-limit-cycle system,
with and without the Theorem-4 period correction.

Layout:
  (A) learned NODE vector field (streamplot)        (B) delta(eps) collapse
  (C) UNCORRECTED sup-error heatmap (P-type)         (D) CORRECTED heatmap (B-type)
"""
from uap import config
import os, numpy as np, torch
from uap.common import harness as H
from uap.demos2d import two_lc as TL

OUT = config.FIGDIR
MOD = config.MODELDIR

sys2, T1, T2 = TL.two_lc_system()
model = H.MLPField(2, 128, 2)
model.load_state_dict(torch.load(os.path.join(MOD, "twolc_w128.pt"))); model.eval()
fh = H.field_np_from_model(model)

# --- localized bumps + re-solve the correction amplitudes on THIS model ------
wb = 0.35
psi = lambda P, rc: np.exp(-0.5*((np.hypot(P[..., 0], P[..., 1]) - rc)/wb)**2)
corrected = lambda a: (lambda P: (1 + a[0]*psi(P, 1.0) + a[1]*psi(P, 2.0))[..., None]*fh(P))
Th1, Th2 = TL.period_from(fh, 1.0), TL.period_from(fh, 2.0)
a = np.array([(Th1-T1)/T1, (Th2-T2)/T2])
for _ in range(8):
    fc = corrected(a)
    p1, p2 = TL.period_from(fc, 1.0), TL.period_from(fc, 2.0)
    r1, r2 = (p1-T1)/T1, (p2-T2)/T2
    if max(abs(r1), abs(r2)) < 1e-6:
        break
    a[0] += (1+a[0])*r1; a[1] += (1+a[1])*r2
fc = corrected(a)
print(f"learned periods {Th1:.5f},{Th2:.5f}; alpha=({a[0]:+.5f},{a[1]:+.5f}); "
      f"corrected {p1:.6f},{p2:.6f}")

# --- IC grid: sup errors, uncorrected vs corrected (cached) ------------------
(xlo, xhi), (ylo, yhi) = sys2.domain
ng, sg = 44, 34
gx = np.linspace(xlo, xhi, ng); gy = np.linspace(ylo, yhi, ng)
XX, YY = np.meshgrid(gx, gy)
SX, SY = np.meshgrid(np.linspace(xlo, xhi, sg), np.linspace(ylo, yhi, sg))
eps = np.logspace(-2.3, np.log10(4), 60)
cache = os.path.join(MOD, "twolc_eps_cache.npz")
if os.path.exists(cache):
    z = np.load(cache)
    sup_b, sup_c, d_b, d_c, F = z["sup_b"], z["sup_c"], z["d_b"], z["d_c"], z["F"]
    print("  loaded eps cache")
else:
    X0 = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    Tend, dt = 200.0, 0.06
    print("  integrating true / uncorrected / corrected ...")
    ts, Yt = H.rk4(sys2.f_np, X0, Tend, dt, save_every=4)
    _,  Yb = H.rk4(fh,        X0, Tend, dt, save_every=4)
    _,  Yc = H.rk4(fc,        X0, Tend, dt, save_every=4)
    sup_b = np.linalg.norm(Yt - Yb, axis=-1).max(axis=0)
    sup_c = np.linalg.norm(Yt - Yc, axis=-1).max(axis=0)
    basin_t = (np.hypot(Yt[-1, :, 0], Yt[-1, :, 1]) > 1.5).astype(int)
    basin_c = (np.hypot(Yc[-1, :, 0], Yc[-1, :, 1]) > 1.5).astype(int)
    valid = np.hypot(X0[:, 0], X0[:, 1]) > 0.25
    print(f"  corrected basin agreement {(basin_t == basin_c)[valid].mean()*100:.2f}%; "
          f"median sup corr {np.median(sup_c[valid]):.3f} vs uncorr {np.median(sup_b[valid]):.3f}")
    d_b = np.array([(sup_b[valid] > e).mean() for e in eps])
    d_c = np.array([(sup_c[valid] > e).mean() for e in eps])
    F = fh(np.stack([SX.ravel(), SY.ravel()], -1)).reshape(sg, sg, 2)
    np.savez(cache, sup_b=sup_b, sup_c=sup_c, d_b=d_b, d_c=d_c, F=F)

# ============================================================ figure
plt, sns = H.setup_style()
import matplotlib.colors as mcolors
fig, ax = plt.subplots(2, 2, figsize=(10.6, 9.7))
LC, SEP, CMAP = H.OKABE_ITO[0], H.OKABE_ITO[2], "rocket"   # blue cycles, green sep

def cycles(a, lw=1.7):
    th = np.linspace(0, 2*np.pi, 240)
    a.plot(np.cos(th), np.sin(th), color=LC, lw=lw)
    a.plot(2*np.cos(th), 2*np.sin(th), color=LC, lw=lw)
    a.plot(1.5*np.cos(th), 1.5*np.sin(th), color=SEP, lw=lw, ls="--")

def heat(a, sup, title):
    pc = a.pcolormesh(XX, YY, sup.reshape(ng, ng), shading="auto", cmap=CMAP,
                      norm=mcolors.LogNorm(vmin=1e-2, vmax=4))
    cycles(a); a.set_aspect("equal"); a.set_xlim(xlo, xhi); a.set_ylim(ylo, yhi)
    a.set_xlabel(r"$x$"); a.set_ylabel(r"$y$"); a.set_title(title)
    cb = fig.colorbar(pc, ax=a, fraction=0.046, pad=0.03)
    cb.set_label(r"$\sup_t\,\|\varphi-\hat\varphi\|$", fontsize=13)

# (A) learned VF streamplot
a = ax[0, 0]
a.streamplot(SX, SY, F[..., 0], F[..., 1],
             color=np.hypot(F[..., 0], F[..., 1]), cmap=CMAP, density=1.1,
             linewidth=0.8, arrowsize=0.7)
cycles(a); a.set_aspect("equal"); a.set_xlim(xlo, xhi); a.set_ylim(ylo, yhi)
a.set_xlabel(r"$x$"); a.set_ylabel(r"$y$")
a.set_title(r"(A) learned NODE field $\hat f$")

# (B) delta(eps)
a = ax[0, 1]
a.semilogx(eps, d_b, color=H.OKABE_ITO[1], lw=2.4, label="uncorrected")
a.semilogx(eps, d_c, color=H.OKABE_ITO[3], lw=2.4, label="corrected")
a.set_xlabel(r"$\varepsilon$")
a.set_ylabel(r"$\delta(\varepsilon)=\mu\{\sup_t\|\varphi-\hat\varphi\|>\varepsilon\}$",
             fontsize=12)
a.set_ylim(-0.03, 1.05)
a.legend(fontsize=12, loc="upper right", frameon=False)
a.set_title(r"(B) period correction collapses $\delta(\varepsilon)$")

# (C) uncorrected heatmap, (D) corrected heatmap
heat(ax[1, 0], sup_b, r"(C) uncorrected: P-type fills both basins")
heat(ax[1, 1], sup_c, r"(D) corrected: only B-type collar at $r=1.5$")

sns.despine(fig=fig); fig.tight_layout()
fig.savefig(os.path.join(OUT, "node_two_lc_epsdelta.pdf"))
fig.savefig(os.path.join(OUT, "node_two_lc_epsdelta.png"))
print("  saved figures/node_two_lc_epsdelta.{pdf,png}")
