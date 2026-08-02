#!/usr/bin/env python3
"""Fine-tune the distilled NODE so its fixed points coincide with the trained
network's. We regress the field to the exact residual field g* globally (tube),
with heavy weight in a neighborhood of the network's fixed points (where g*=0, so
this drives the NODE's field to vanish there too), plus a windowed trajectory term
to preserve the rollout fit. Overwrites node_ctd_3bff_tube.pt. Run in CtDEnv."""
import os, time, numpy as np, torch
from uap.flipflop import fit_node_tube as F

MODELDIR = F.MODELDIR
N, NU, sig, gstar, Hf, Uf, H, U = F.N, F.NU, F.sig, F.gstar, F.Hf, F.Uf, F.H, F.U
f = F.Field(384, 3)
f.load_state_dict(torch.load(os.path.join(MODELDIR, "node_ctd_3bff_tube.pt")))

d = np.load(os.path.join(MODELDIR, "ctd_3bff_fps.npz"))
Xfp = torch.tensor(d["xs"][(d["q"] < 1e-4) & d["stab"].astype(bool)], dtype=torch.float64)
print(f"anchoring to {Xfp.shape[0]} network fixed points", flush=True)
P = Hf.shape[0]; Pu = Uf.shape[0]; B, T = H.shape[0], H.shape[1]
opt = torch.optim.Adam(f.parameters(), 3e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3500)
gen = torch.Generator().manual_seed(0); z0 = torch.zeros(1, NU); t0 = time.time()
for ep in range(3500):
    ix = torch.randint(0, P, (1024,), generator=gen); iu = torch.randint(0, Pu, (1024,), generator=gen)
    x = Hf[ix] + (0.4 * torch.rand(1024, 1, generator=gen)) * sig * torch.randn(1024, N, generator=gen)
    loss_tube = ((f(x, Uf[iu]) - gstar(x, Uf[iu])) ** 2).mean()
    jf = torch.randint(0, Xfp.shape[0], (768,), generator=gen)
    xf = Xfp[jf] + 0.15 * torch.randn(768, N, generator=gen)
    zf = z0.expand(768, NU)
    loss_fp = ((f(xf, zf) - gstar(xf, zf)) ** 2).mean()
    L = 48
    b = torch.randint(0, B, (96,), generator=gen); t = torch.randint(0, T - L - 1, (96,), generator=gen)
    Hwin = H[b[:, None], t[:, None] + torch.arange(L + 1)]; Uwin = U[b[:, None], t[:, None] + torch.arange(L)]
    xx = Hwin[:, 0]; preds = []
    for k in range(L):
        xx = xx + f(xx, Uwin[:, k]); preds.append(xx)
    loss_traj = ((torch.stack(preds, 1) - Hwin[:, 1:]) ** 2).mean()
    loss = loss_tube + 5.0 * loss_fp + loss_traj
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(f.parameters(), 5.0); opt.step(); sch.step()
    if (ep + 1) % 500 == 0:
        print(f"ep{ep+1} tube {loss_tube.item():.2e} fp {loss_fp.item():.2e} traj {loss_traj.item():.2e}", flush=True)
torch.save(f.state_dict(), os.path.join(MODELDIR, "node_ctd_3bff_tube.pt"))
print(f"fine-tuned {time.time()-t0:.0f}s -> saved", flush=True)
