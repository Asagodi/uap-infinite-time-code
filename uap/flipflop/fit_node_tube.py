#!/usr/bin/env python3
"""Distil the CtD trained 3BFF GRU into an INDEPENDENT MLP Neural ODE, done right.

The recorded "latents" are a stochastic realization (per-step state noise), so no
deterministic field reproduces them. We instead take the network's DETERMINISTIC
dynamical system directly: the GRU cell defines
    h_{t+1} = cell(u_t, h_t),
which is exactly forward-Euler (dt=1) of the ODE  xdot = g*(x,u), g*(x,u)=cell(u,x)-x.
We generate the deterministic reference trajectory ourselves (iterate the cell from
the real first states, driven by the real input sequences), so the exact field
reproduces it with delta=0 by construction. Then we regress an INDEPENDENT MLP field
onto g* over a TUBE around the trajectories (states + perturbations), roll it out with
Euler dt=1, and score eps-delta on the network's OWN readout -- no refit, no held-out,
the paper's metric. Run in CtDEnv.
"""
import os, time
from uap import config
config.chdir_ctd()   # chdir into the CtD toolkit (set CTD_TOOLKIT_DIR in .env to re-train)
import numpy as np, torch, torch.nn as nn
torch.set_default_dtype(torch.float64); torch.set_num_threads(6)
from ctd.comparison.analysis.tt.tt import Analysis_TT

FIGDIR = config.FIGDIR
MODELDIR = config.MODELDIR
BASE = "content/trained_models/task-trained/20260712_3bff"
RUN = "latent_size=64_num_workers=0_num_workers=0_max_epochs=500_seed=0"
tt = Analysis_TT("s0", os.path.join(BASE, RUN) + "/", use_train_dm=True)
m = tt.model
cell = m.cell.eval()                                          # GRUCell(input=u, hidden=x), float32
Wout = m.readout.weight.detach().to(torch.float64)
bout = m.readout.bias.detach().to(torch.float64)
def readout(X): return X @ Wout.T + bout                      # network's OWN readout, fixed

@torch.no_grad()                                              # exact continuous field (cell is float32)
def gstar(x, u): return (cell(u.float(), x.float()) - x.float()).to(torch.float64)

# real inputs + a set of real starting states, then generate the DETERMINISTIC reference
outn = tt.get_model_outputs_noiseless(phase="all")
Hrec = outn["latents"].detach().to(torch.float64)             # recorded (noisy) latents -> ICs + tube seeds
_i, U, _t = tt.get_model_inputs(phase="all")
U = torch.as_tensor(U, dtype=torch.float64)                   # (B,T,NU)
B, T, N = Hrec.shape; NU = U.shape[-1]

@torch.no_grad()
def rollout(f, h0, useq):                                     # Euler dt=1, T states incl. x0
    x = h0.clone(); xs = [x.clone()]
    for t in range(useq.shape[1] - 1):
        x = x + f(x, useq[:, t]); xs.append(x.clone())
    return torch.stack(xs, 1)

H = rollout(gstar, Hrec[:, 0], U)                            # deterministic GRU latent (B,T,N)
Hf = H.reshape(-1, N); Uf = U[:, :-1].reshape(-1, NU); P = Hf.shape[0]
Yg = readout(H); THR = Yg.reshape(-1, 3).mean(0).numpy()
sig = Hf.std(0)                                              # per-dim latent scale for the tube

Xexact = rollout(gstar, H[:20, 0], U[:20])
print(f"[sanity] exact-field rollout vs deterministic GRU: sup={((H[:20]-Xexact).norm(dim=-1).max()):.2e} "
      f"(must be ~0)  |g*|~{gstar(Hf[:4000],Uf[:4000] if P==Uf.shape[0] else Uf[:4000]).norm(dim=1).mean():.3f}  "
      f"thr={np.round(THR,2)}", flush=True)


class Field(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        Lyr, d = [], N + NU
        for _ in range(depth):
            Lyr += [nn.Linear(d, width), nn.Tanh()]; d = width
        Lyr += [nn.Linear(d, N)]
        self.net = nn.Sequential(*Lyr)
    def forward(self, x, u): return self.net(torch.cat([x, u], -1))


def train(width, depth, epochs, bs=4096, lr=1.5e-3, seed=0, tube=0.6):
    torch.manual_seed(seed); f = Field(width, depth)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    gen = torch.Generator().manual_seed(seed); t0 = time.time()
    Pu = Uf.shape[0]
    for ep in range(epochs):
        ix = torch.randint(0, P, (bs,), generator=gen)
        iu = torch.randint(0, Pu, (bs,), generator=gen)
        x0 = Hf[ix]; u = Uf[iu]                                # decorrelate state and input over the tube
        s = tube * torch.rand(bs, 1, generator=gen)            # random-scale Gaussian shell
        x = x0 + s * sig * torch.randn(bs, N, generator=gen)
        tgt = gstar(x, u)
        loss = ((f(x, u) - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 5.0); opt.step(); sch.step()
        if (ep + 1) % max(1, epochs // 8) == 0:
            print(f"    ep{ep+1} field-MSE {loss.item():.3e}", flush=True)
    print(f"    trained {time.time()-t0:.0f}s ({sum(p.numel() for p in f.parameters())} params)", flush=True)
    return f


def train_traj(f, epochs, bs=128, lr=3e-4, seed=1, Lcur=(16, 32, 64, 96), tube=0.25, wreg=0.3):
    """Trajectory fine-tuning: unroll the field (Euler dt=1) over growing windows and
    match the clean deterministic reference (BPTT), plus a small tube-regression anchor
    to g* so off-path states stay controlled. Directly fights Euler error accumulation."""
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    gen = torch.Generator().manual_seed(seed); t0 = time.time(); Pu = Uf.shape[0]
    for ep in range(epochs):
        L = Lcur[min(len(Lcur) - 1, int(len(Lcur) * ep / epochs))]
        b = torch.randint(0, B, (bs,), generator=gen)
        t = torch.randint(0, T - L - 1, (bs,), generator=gen)
        Hwin = H[b[:, None], t[:, None] + torch.arange(L + 1)]
        Uwin = U[b[:, None], t[:, None] + torch.arange(L)]
        x = Hwin[:, 0]; preds = []
        for k in range(L):
            x = x + f(x, Uwin[:, k]); preds.append(x)
        traj = ((torch.stack(preds, 1) - Hwin[:, 1:]) ** 2).mean()
        # tube anchor to the exact field
        ix = torch.randint(0, P, (bs * 4,), generator=gen); iu = torch.randint(0, Pu, (bs * 4,), generator=gen)
        xa = Hf[ix] + tube * torch.rand(bs * 4, 1, generator=gen) * sig * torch.randn(bs * 4, N, generator=gen)
        anch = ((f(xa, Uf[iu]) - gstar(xa, Uf[iu])) ** 2).mean()
        loss = traj + wreg * anch
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 5.0); opt.step(); sch.step()
        if (ep + 1) % max(1, epochs // 8) == 0:
            print(f"    ft ep{ep+1} traj-MSE(L={L}) {traj.item():.3e} anchor {anch.item():.3e}", flush=True)
    print(f"    fine-tuned {time.time()-t0:.0f}s", flush=True)
    return f


def evaluate(f, nb=None):
    nb = B if nb is None else min(nb, B)
    xs = rollout(f, H[:nb, 0], U[:nb])
    yg = readout(H[:nb]).numpy(); yn = readout(xs).numpy()
    sup = np.linalg.norm(yg - yn, axis=-1).max(1)             # sup_t output error, built-in readout
    bit = float((np.sign(yn - THR) == np.sign(yg - THR)).mean())
    latsup = (H[:nb] - xs).norm(dim=-1).max(1).values.mean().item()
    return dict(sup=sup, bit=bit, latsup=latsup)


def node_fps(f):
    z = torch.zeros(1, NU)
    X = Hf[torch.randint(0, P, (600,))].clone().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=2e-2)
    for _ in range(2500):
        opt.zero_grad(); g = f(X, z.expand(X.shape[0], NU))
        (g ** 2).sum(1).mean().backward(); opt.step()
    with torch.no_grad():
        sp = f(X, z.expand(X.shape[0], NU)).norm(dim=1).numpy()
        ro = readout(X.detach()).numpy()
    good = sp < 1e-3
    patt = np.unique((ro[good] > THR).astype(int), axis=0) if good.sum() else np.zeros((0, 3))
    return len(patt), ro[good]


def main():
    # (width, depth, tube-pretrain epochs, trajectory-fine-tune epochs)
    plan = [(448, 3, 4500, 4000), (640, 4, 6000, 5500)]
    best = None; best_d = 1e9
    for width, depth, pep, fep in plan:
        print(f"\n=== field width={width} depth={depth} pretrain={pep} finetune={fep} ===", flush=True)
        f = train(width, depth, pep)                          # tube-regression pretrain
        f = train_traj(f, fep)                                # trajectory fine-tune (BPTT)
        ev = evaluate(f); nfp, roF = node_fps(f)
        d = ev['sup']; d03 = float((d > 0.3).mean())
        print(f"  built-in readout: bit-acc {ev['bit']*100:.2f}%  latent-sup {ev['latsup']:.3f}  "
              f"median-sup {np.median(d):.3f}  delta(0.1)={(d>0.1).mean():.3f} "
              f"delta(0.3)={d03:.3f}  node-FPs {nfp}/8", flush=True)
        if d03 < best_d:
            best_d = d03; best = (f, ev, nfp, roF, width)
        if d03 < 0.15 and nfp == 8:
            print("  >>> GOOD eps-delta FIT reached.", flush=True); break

    f, ev, nfp, roF, width = best
    os.makedirs(MODELDIR, exist_ok=True)
    torch.save(f.state_dict(), os.path.join(MODELDIR, "node_ctd_3bff_tube.pt"))
    sup = ev['sup']
    import matplotlib.font_manager as fm, matplotlib.pyplot as plt
    for _f in fm.findSystemFonts(fontpaths=[os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")]):
        try: fm.fontManager.addfont(_f)
        except Exception: pass
    plt.rcParams.update({"font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 14,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "savefig.dpi": 300})
    OK = ["#0072B2", "#D55E00"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    eps = np.logspace(-2.3, 0.3, 60)
    ax[0].plot(eps, [(sup > e).mean() for e in eps], color=OK[0], lw=2.6)
    ax[0].axvline(np.median(sup), color="0.6", lw=1.4, ls="--")
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\varepsilon$"); ax[0].set_ylabel(r"$\delta(\varepsilon)$")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_title(f"NODE vs GRU output (own readout)\nbit-acc {ev['bit']*100:.1f}%")
    c3 = np.where(roF[:, 2] > THR[2], OK[0], OK[1])
    ax[1].scatter(roF[:, 0], roF[:, 1], c=c3, s=42, alpha=0.85, zorder=2)
    ax[1].axvline(THR[0], color="0.85", lw=1); ax[1].axhline(THR[1], color="0.85", lw=1)
    ax[1].set_xlabel("readout bit 1"); ax[1].set_ylabel("readout bit 2")
    ax[1].set_title(f"NODE autonomous fixed points\n{nfp}/8 memory states")
    fig.tight_layout()
    for a, Lab in zip(ax, "AB"):
        a.text(-0.17, 1.05, f"({Lab})", transform=a.transAxes, fontsize=16, fontweight="bold")
    fig.savefig(os.path.join(FIGDIR, "node_ctd_3bff_fit.pdf"))
    fig.savefig(os.path.join(FIGDIR, "node_ctd_3bff_fit.png"))
    print(f"\nFINAL width={width} bit-acc={ev['bit']*100:.2f}% delta(0.3)={(sup>0.3).mean():.4f} "
          f"delta(0.1)={(sup>0.1).mean():.4f} node-fps={nfp}/8  saved figure", flush=True)


if __name__ == "__main__":
    main()
