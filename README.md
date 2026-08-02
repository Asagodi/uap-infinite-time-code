# Infinite-time universal approximation: experiment code

Code to reproduce the experiments in

> **Universal Approximation Theorems for Dynamical Systems with Infinite-Time Horizon Guarantees**
> Abel Ságodi and Il Memming Park (2026). [arXiv:2602.08640](https://arxiv.org/abs/2602.08640)

The paper proves that Neural ODEs can approximate structurally stable (Morse-Smale)
dynamical systems over the whole infinite-time horizon in an ε-δ sense (trajectory
error below ε for all initial conditions except a set of measure below δ), and
classifies the ways an approximation can fail into three modes: **B-type** (basin
mismatch), **P-type** (phase drift on limit cycles), and **D-type** (discretization
of a continuous attractor). This repository generates the empirical figures, on
trained Neural ODEs and on real task-trained networks.

## Layout

```
uap/                    the package (run modules with `python -m uap.<group>.<name>`)
├── config.py           paths, configurable via environment / .env
├── common/harness.py   target systems, MLP field, C^1 training, RK4, eps-delta, style
├── demos2d/            2D / low-dim demos: Duffing, Van der Pol, ring, two-cycle,
│                       8 fixed points, and the bistable panels
├── flipflop/           3-bit flip-flop (distil a Neural ODE from a CtD-trained network)
├── ring_hd/            head-direction ring (real BTCA network), D-type discretization
└── scaling/            budget / width scaling and period-correction checks
models/                 bundled checkpoints and caches (see models/PROVENANCE.md)
figures/                the paper figures (reference PDF + PNG); scripts write here too
requirements-iam.txt    dependencies for the self-contained figures
requirements-ctd.txt    dependencies for the flip-flop figures (external toolkit)
environment.yml         conda environment for the self-contained figures
pyproject.toml          installable package (pip install -e .)
.env.example            external-toolkit paths (only needed to re-train from scratch)
```

## Setup

Two environments are used; most figures need only the first.

**A. Self-contained figures.** Everything except the 3-bit flip-flop and the real
head-direction ring:

```bash
conda env create -f environment.yml && conda activate uap-infinite-time
pip install -e .          # or: pip install -r requirements-iam.txt
```

**B. Flip-flop figures (optional).** Figures 4C, 4D and 7(F–H) distil a Neural ODE
from a task-trained 3-bit flip-flop network and require the external
[Computation-through-Dynamics toolkit](https://github.com/snel-repo/ComputationThroughDynamicsToolkit).
Install it into its own Python 3.10 environment, then:

```bash
pip install -r requirements-ctd.txt      # on top of the toolkit's environment
cp .env.example .env                      # set CTD_TOOLKIT_DIR (and BTCA_CHECKPOINTS_DIR)
```

## Quick start

Run modules with `python -m` from the repository root; output goes to `figures/`.

```bash
python -m uap.demos2d.run_vanderpol         # Fig 5  (P-type, Van der Pol)
python -m uap.demos2d.make_duffing_panels   # Fig 4A,B  (B-type, Duffing)
python -m uap.demos2d.two_lc_eps            # Fig 6  (two limit cycles)
python -m uap.demos2d.viz_8fp              # Fig 7A–C  (8 fixed points)
python -m uap.demos2d.run_multi_arch_8fp   # Fig 7D–E  (three architectures)
```

The bundled checkpoints in `models/` mean these run in seconds without re-training.

## Reproducing each figure

| Fig | Image | Command | External model |
|-----|-------|---------|----------------|
| 1 | `hyp_land_rnns_fmp` | schematic (hand-drawn, not scripted) | — |
| 2 | `separatrices` | schematic (hand-drawn, not scripted) | — |
| 3a | `bistable_trajectories` | `python -m uap.demos2d.mkfig_bi_traj` | — |
| 3b | `bistable_errors` | `python -m uap.demos2d.mkfig_bi_error` | — |
| 4A,B | `node_duffing_single`, `node_duffing_epsdelta` | `python -m uap.demos2d.make_duffing_panels` | — |
| 4C | `node_3bff_basins` | `python -m uap.flipflop.make_3bff_basins` | CtD flip-flop |
| 4D | `node_3bff_epsdelta` | `python -m uap.flipflop.make_3bff_epsdelta` | CtD flip-flop |
| 5 | `node_vanderpol_ptype` | `python -m uap.demos2d.run_vanderpol` | — |
| 6 | `node_two_lc_epsdelta` | `python -m uap.demos2d.two_lc_eps` | — |
| 7A–C | `node_8fp_hero` | `python -m uap.demos2d.viz_8fp` | — |
| 7D–E | `node_multiarch_8fp` | `python -m uap.demos2d.run_multi_arch_8fp` | — |
| 7F–H | `node_ctd_3bff_fit` | `python -m uap.flipflop.refresh_ctd_fig` | CtD flip-flop |
| 8 | `node_ca_combined` | `python -m uap.ring_hd.make_ca_combined` | BTCA ring (bottom row) |

**External model** names a trained network that must be obtained separately to
*re-generate* the figure; it otherwise reproduces from the bundled checkpoint or
cache in `models/`. Figures 1 and 2 are hand-drawn schematics and are not produced
by a script (the PDFs live with the manuscript).

## Re-training from scratch (optional)

The bundled checkpoints reproduce the figures without training. To regenerate:

- **2D targets.** Delete the relevant `models/*.pt`; the demo modules retrain on the
  next run.
- **Flip-flop (Figs 4C, 4D, 7F–H).** With `CTD_TOOLKIT_DIR` set in `.env`, run
  `uap.flipflop.analyze_ctd_3bff` (fixed points), then `uap.flipflop.fit_node_tube`
  (distil the Neural ODE), optionally `uap.flipflop.fine_tune_fp`, then the figure
  modules. These import the live flip-flop network, so the toolkit is required even
  to redraw.
- **BTCA ring (Fig 8, bottom row).** With `BTCA_CHECKPOINTS_DIR` set, run
  `uap.ring_hd.fca_ring_btca` to build `models/btca_ring_cache.npz`, then
  `uap.ring_hd.make_ca_combined`.

## Supplementary (`uap.scaling`)

Backing the appendix budget/width scaling (η₀ ~ εδ, width ~ (εδ)^(-1/p)) and the
period-correction claims, self-contained, console output plus supplementary plots:
`budget_width`, `run_sizelaw`, `period_vs_width`, `pc_precise`, `period_correction`.

## Citation

```bibtex
@article{sagodi2026uap,
  title   = {Universal Approximation Theorems for Dynamical Systems with Infinite-Time Horizon Guarantees},
  author  = {S{\'a}godi, Abel and Park, Il Memming},
  year    = {2026},
  journal = {arXiv preprint arXiv:2602.08640}
}
```

## License

MIT. See [LICENSE](LICENSE). Bundled checkpoints derived from external models are
documented in [models/PROVENANCE.md](models/PROVENANCE.md).
