# === Final 2×3 Visualization of All Curves (target vs. your best trace) ===
# Usage:
# 1) Set SUBMISSION_NPY to your saved file (np.save(..., submission, allow_pickle=True))
# 2) Run. It saves figures and prints a small metrics table.

import numpy as np
import matplotlib.pyplot as plt
from LINKS.Kinematics import MechanismSolver
from LINKS.Geometry import CurveEngine
from LINKS.Optimization import Tools

# ---- 1) Point to your final submission .npy ----
SUBMISSION_NPY = "CP1_submission_SM_BA.npy"  # <- change me

# ---- 2) Load data & tools ----
submission = np.load(SUBMISSION_NPY, allow_pickle=True).item()
target_curves = np.load("target_curves.npy")

TOOLS = Tools(device="cpu")
TOOLS.compile()
solver = MechanismSolver(device="cpu")
curve_engine = CurveEngine(device="cpu", normalize_scale=False)  # scale must remain meaningful

# ---- 3) Helper: evaluate a list of mechs for one problem, return idx of best distance (feasible only) ----
def pick_best_by_distance(mech_list, target_curve):
    if not mech_list:
        return None, None, None
    x0s, edges, fixeds, motors, tjs = [],
