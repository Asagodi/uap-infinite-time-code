"""Central configuration for the infinite-time UAP code release.

Paths resolve from environment variables (optionally via a local ``.env`` file),
falling back to defaults relative to this file. The bundled checkpoints in
``models/`` reproduce every self-contained figure with no external toolkit. To
RE-TRAIN the CtD 3-bit-flip-flop distillation or re-run the BTCA ring analysis
from scratch, copy ``.env.example`` to ``.env`` and set the two external paths
below (see the README).
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))   # the uap/ package directory
ROOT = os.path.dirname(HERE)                          # repository root

# Output and checkpoint directories (relative to the repo root by default).
FIGDIR = os.environ.get("UAP_FIGDIR", os.path.join(ROOT, "figures"))
MODELDIR = os.environ.get("UAP_MODELDIR", os.path.join(ROOT, "models"))
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

# External toolkits, needed only to RE-TRAIN from scratch (see the README).
# Checkout of the Computation-through-Dynamics toolkit (for the 3-bit flip-flop).
CTD_TOOLKIT_DIR = os.environ.get("CTD_TOOLKIT_DIR", "")
# Directory of BTCA recttanh head-direction checkpoints (result_dict_*.pickle).
BTCA_CHECKPOINTS_DIR = os.environ.get("BTCA_CHECKPOINTS_DIR", "")


def chdir_ctd():
    """chdir into the CtD toolkit if configured; its loaders use relative paths."""
    if CTD_TOOLKIT_DIR:
        os.chdir(CTD_TOOLKIT_DIR)
    return CTD_TOOLKIT_DIR
