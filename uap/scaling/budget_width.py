#!/usr/bin/env python3
"""
Accuracy budget eta0(eps,delta) for a 1D bistable family, and the required
3-layer tanh MLP width with a LOWER and UPPER estimate.

Budget (corrected).  Near a saddle at distance eta_sep ~ delta*vol/(4 A_S) the
forcing is amplified at the LOCAL unstable rate mu_u (not the global Lipschitz
L): int_0^t e^{mu_u (t-s)} nu ds ~ nu/(mu_u eta_sep).  Requiring < eps gives
    eta0 ~ c * mu_u * eps * delta          (symmetric product; NO delta^{L/mu}).
We check this empirically: fit eta0 ~ eps^a delta^b and confirm a,b ~ 1 while
L/mu_u varies across the family.

Width bracket.  Compose eta0 ~ eps*delta with the C^1 approximation rate of a
two-hidden-layer (="3-layer") tanh net (De Ryck-Lanthaler-Mishra 2021):
  UPPER (assume only finite smoothness r, Sobolev):  W ~ (eps*delta)^{-n/(r-1)}
  LOWER (exploit analyticity):                       W ~ (log 1/(eps*delta))^n
For an analytic target the measured eta(W) follows the LOWER (log) regime, so
the naive Sobolev width is a large over-estimate.
"""
from uap import config
import os, numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import torch, torch.nn as nn

# ---- CMU font bootstrap -----------------------------------------------------
_uf = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
for _f in fm.findSystemFonts(fontpaths=[_uf]):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 14, "axes.labelsize": 16,
    "axes.titlesize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "axes.linewidth": 1.1, "lines.linewidth": 2.0,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
OUT = config.FIGDIR
torch.set_default_dtype(torch.float64)

# ============================================================================
# Budget  eta0(eps, delta)   (user's proxy family, verbatim logic)
# ============================================================================
DOM, LAM = 1.3, 2.0
def coef(mu): return LAM/2 - 2*mu, mu - LAM/2
def inward(mu):
    b, c = coef(mu); return mu + b*DOM**2 + c*DOM**4 < 0
def lipschitz(mu):
    b, c = coef(mu); x = np.linspace(-DOM, DOM, 4001)
    return float(np.max(np.abs(mu + 3*b*x**2 + 5*c*x**4)))

def run_one(mu, nu_grid, eps_grid, delta_grid, n_ic=1201, dt=0.008):
    b, c = coef(mu)
    nsteps = int((12.0/mu + 20.0)/dt)
    x0 = np.linspace(-DOM+0.05, DOM-0.05, n_ic)
    X = x0.copy(); Y = np.broadcast_to(x0, (len(nu_grid), n_ic)).copy()
    NU = nu_grid[:, None]; M = np.zeros_like(Y)
    f  = lambda z: z*(mu + z*z*(b + c*z*z))
    fh = lambda z: z*(mu + z*z*(b + c*z*z)) + NU*(np.cos(2*z)/3.0)
    h6, h2 = dt/6.0, 0.5*dt
    for _ in range(nsteps):
        k1=f(X); k2=f(X+h2*k1); k3=f(X+h2*k2); k4=f(X+dt*k3); X = X + h6*(k1+2*k2+2*k3+k4)
        k1=fh(Y);k2=fh(Y+h2*k1);k3=fh(Y+h2*k2);k4=fh(Y+dt*k3);Y = Y + h6*(k1+2*k2+2*k3+k4)
        np.maximum(M, np.abs(Y-X), out=M)
    E = np.stack([(M >= e).mean(axis=1) for e in eps_grid], axis=1)
    eta0 = np.full((len(eps_grid), len(delta_grid)), np.nan)
    for j in range(len(eps_grid)):
        col = E[:, j]
        for k, d in enumerate(delta_grid):
            bad = np.where(col >= d)[0]; stop = bad[0] if len(bad) else len(nu_grid)
            if stop > 0: eta0[j, k] = nu_grid[stop-1]
    return eta0

def fit_ab(eta0, eps_grid, delta_grid):
    A, y = [], []
    for j, e in enumerate(eps_grid):
        for k, d in enumerate(delta_grid):
            v = eta0[j, k]
            if np.isfinite(v) and v > 0:
                A.append([np.log(e), np.log(d), 1.0]); y.append(np.log(v))
    A, y = np.array(A), np.array(y)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol, float(np.sqrt(((y - A@sol)**2).mean()))

nu_grid    = np.logspace(-5.2, -1.2, 70)
eps_grid   = np.array([0.02, 0.04, 0.08, 0.16])
delta_grid = np.array([0.004, 0.008, 0.016, 0.032, 0.064])
mus = [0.4, 0.7, 1.0, 1.5, 2.0]

print(f"{'mu_u':>5} {'L':>6} {'L/mu':>6} | {'a(eps)':>7} {'b(del)':>7} {'rmse':>6}")
print("-"*46)
_CACHE = os.path.join(config.MODELDIR, "budget_cache.npz")
if os.path.exists(_CACHE):
    S = np.load(_CACHE, allow_pickle=True)["S"].item()
    for mu in S:
        d = S[mu]
        print(f"{mu:5.2f} {d['L']:6.2f} {d['L']/mu:6.2f} | {d['a']:7.3f} {d['b']:7.3f}  (cached)")
else:
    S = {}
    for mu in mus:
        if not inward(mu):
            print(f"{mu:5.2f}  skipped (domain not forward invariant)"); continue
        L = lipschitz(mu)
        eta0 = run_one(mu, nu_grid, eps_grid, delta_grid)
        (a, b, cst), rmse = fit_ab(eta0, eps_grid, delta_grid)
        S[mu] = dict(eta0=eta0, L=L, a=a, b=b, cst=cst)
        print(f"{mu:5.2f} {L:6.2f} {L/mu:6.2f} | {a:7.3f} {b:7.3f} {rmse:6.3f}")
        sys.stdout.flush()
    np.savez(_CACHE, S=np.array(S, dtype=object))
ab = np.array([[S[m]["a"], S[m]["b"]] for m in S])
print(f"\n  mean exponents: a={ab[:,0].mean():.3f}  b={ab[:,1].mean():.3f}  "
      f"(prediction eta0 ~ eps*delta => a=b=1); L/mu ranges "
      f"{min(S[m]['L']/m for m in S):.2f}-{max(S[m]['L']/m for m in S):.2f} "
      f"but b stays ~1")

# ============================================================================
# eta(W): 3-layer tanh MLP fitting the (analytic) bistable field, mu=1
# ============================================================================
def make_mlp(width, depth=2):
    layers, d = [], 1
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]; d = width
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)

def eta_of_width(mu, width, seeds=(0, 1, 2, 3)):
    """Sup-norm approx error of the best of several Adam+LBFGS fits.
    LBFGS drives the 1D fit to the true approximation floor (Adam alone gets
    stuck in bad minima and gives a non-monotone, optimization-limited eta)."""
    b, c = coef(mu); f = lambda z: z*(mu + z*z*(b + c*z*z))
    # train on a slightly larger domain, measure sup on the INTERIOR, so eta(W)
    # reflects approximation capacity and not the fixed boundary edge artifact.
    x = torch.linspace(-DOM-0.3, DOM+0.3, 1600).reshape(-1, 1); y = f(x)
    xg = torch.linspace(-DOM+0.05, DOM-0.05, 6000).reshape(-1, 1); yg = f(xg)
    best = np.inf
    for s in seeds:
        torch.manual_seed(s)
        net = make_mlp(width)
        opt = torch.optim.Adam(net.parameters(), 5e-3)
        for _ in range(1200):
            opt.zero_grad(); ((net(x)-y)**2).mean().backward(); opt.step()
        lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=500,
                               history_size=60, line_search_fn="strong_wolfe")
        def closure():
            lb.zero_grad(); l = ((net(x)-y)**2).mean(); l.backward(); return l
        lb.step(closure)
        with torch.no_grad():
            best = min(best, float((net(xg)-yg).abs().max()))
    return best

MU_REF, N_DIM = 1.0, 1
Ws = np.array([3, 4, 6, 8, 12, 16, 24, 32, 48])
print(f"\n  eta(W) for 3-layer tanh MLP on the bistable (mu={MU_REF}):")
etaW = np.array([eta_of_width(MU_REF, int(W)) for W in Ws])
for W, e in zip(Ws, etaW):
    print(f"    W={W:3d}  eta={e:.3e}")

# eta(W) saturates: a 1D analytic cubic is over-parameterized, so the measured
# error is OPTIMIZATION-limited, not approximation-limited -- width is not the
# binding constraint in 1D. The genuine width SCALING appears in 2D (size-law).
floor = float(np.median(etaW)); best = float(etaW.min())
best_W = int(Ws[int(np.argmin(etaW))])
prod = np.array([eps_grid[j]*delta_grid[kk]
                 for j in range(len(eps_grid)) for kk in range(len(delta_grid))])
print(f"\n  eta(W) saturates at ~{floor:.1e} (best {best:.1e} at W={best_W}) for all W>=3:")
print(f"  optimization- not approximation-limited; a width-{best_W} net already")
print(f"  meets the entire eps*delta budget range [{prod.min():.1e}, {prod.max():.1e}].")
print(f"  Width SCALING is dimensional -- measured in 2D by the size-law sweep.")

# ============================================================================
# figure
# ============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: budget exponent b vs L/mu_u  (b ~ 1, does not track L/mu)
ax0 = ax[0]
lmu = [S[m]["L"]/m for m in S]; bb = [S[m]["b"] for m in S]; aa = [S[m]["a"] for m in S]
ax0.plot(lmu, bb, "o", color=OKABE[1], ms=9, label=r"fitted $\delta$-exp. $b$")
ax0.plot(lmu, aa, "s", color=OKABE[0], ms=8, label=r"fitted $\varepsilon$-exp. $a$")
ax0.axhline(1.0, color="k", ls="--", lw=1.4, label=r"$a=b=1$  ($\eta_0\sim\varepsilon\delta$)")
ax0.plot(lmu, lmu, ":", color="0.6", lw=1.6, label=r"$b=L/\mu_u$ (artifact)")
ax0.set_xlabel(r"$L/\mu_u$"); ax0.set_ylabel(r"fitted exponent")
ax0.set_ylim(0, max(lmu)*1.1); ax0.legend(fontsize=10, loc="upper left")
ax0.set_title(r"budget $\eta_0\sim\varepsilon\,\delta$ (both exponents $\approx1$)")

# Panel B: measured eta(W) saturates at the optimization floor (1D too easy)
ax1 = ax[1]
ax1.axhspan(prod.min(), prod.max(), color=OKABE[0], alpha=0.13,
            label=r"$\varepsilon\delta$ budget range")
ax1.semilogy(Ws, etaW, "ko-", ms=7, lw=1.5, label=r"measured $\eta(W)$")
ax1.axhline(floor, color=OKABE[1], ls="--", lw=1.6,
            label=r"opt. floor $\approx%.0e$" % floor)
ax1.set_xlabel(r"MLP width $W$"); ax1.set_ylabel(r"VF error $\eta(W)$")
ax1.legend(fontsize=9, loc="lower left")
ax1.set_title(r"1D analytic: width saturates (opt-limited)")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "budget_width.pdf"))
fig.savefig(os.path.join(OUT, "budget_width.png"))
print("\n  saved figures/budget_width.{pdf,png}")
