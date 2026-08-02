# Bundled checkpoints and caches

## Self-trained in this repo
Trained by the scripts here on synthetic targets; regenerate by deleting the file
and re-running the owning script.

| File | Trained by | Target |
|------|------------|--------|
| `duffing_w64_d2.pt` | `run_duffing.py` | damped Duffing double well |
| `vanderpol_w64_d2.pt` | `run_vanderpol.py` | Van der Pol limit cycle |
| `ring_w64_d2.pt` | `run_ring.py` | ring of equilibria |
| `mlp_8fp_viz.pt` | `viz_8fp.py` | 3D eight-fixed-point cube |
| `twolc_w128.pt` | `two_lc.py` | two coaxial limit cycles |
| `sweep_{duffing,ring,vanderpol}_w*.pt` | `run_sizelaw.py` | width sweep |
| `cache_multiarch.pkl` | `run_multi_arch_8fp.py` | 3-architecture 8-FP results |
| `budget_cache.npz`, `sizelaw_cache.json`, `twolc_eps_cache.npz` | budget / size-law / two-LC scripts | analysis caches |

## Derived from external trained models
These are distillations or analyses of models trained elsewhere. They are bundled
so the corresponding figures reproduce, but re-generating them requires the external
model (see the repo README).

| File | Produced by | External source |
|------|-------------|-----------------|
| `node_ctd_3bff_tube.pt` | `fit_node_tube.py` | Neural ODE distilled from a task-trained 3-bit flip-flop GRU in the Computation-through-Dynamics toolkit (Driscoll et al.-style task-trained model) |
| `ctd_3bff_fps.npz` | `analyze_ctd_3bff.py` | fixed points of that same CtD flip-flop GRU |
| `btca_ring_cache.npz` | `fca_ring_btca.py` | ring-drift / settling analysis of a BTCA recttanh head-direction RNN (Sagodi et al.) |
