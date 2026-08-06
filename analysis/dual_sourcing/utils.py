"""
Shared utilities for dual sourcing post-processing scripts.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# -------------------------------------------------------
# IO paths
# -------------------------------------------------------

# Adjust this to your local IO path
IO_PATH = "C:/DynaPlex_IO/IO_DynaPlex/dual_sourcing"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output")

def get_results_path(filename):
    return os.path.join(IO_PATH, "evaluation", filename)

def get_tuning_path():
    return os.path.join(IO_PATH, "tuning", "tuned_heuristic_params.json")

def get_output_path(filename):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    return os.path.join(OUTPUT_PATH, filename)

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_benchmark_results():
    return load_json(get_results_path("benchmark_results.json"))

def load_convergence_results(action_repr):
    return load_json(get_results_path(f"convergence_{action_repr}.json"))

def load_parameter_results(parameter):
    return load_json(get_results_path(f"parameter_{parameter}.json"))

def load_horizon_results():
    return load_json(get_results_path("horizon_results.json"))

# -------------------------------------------------------
# Computations
# -------------------------------------------------------

def compute_gap(policy_cost, benchmark_cost):
    """Relative gap in percent: negative means policy beats benchmark."""
    if benchmark_cost == 0:
        return float("nan")
    return (policy_cost - benchmark_cost) / benchmark_cost * 100.0

# -------------------------------------------------------
# Plot styling — thesis ready
# -------------------------------------------------------

# Policy colors — consistent across all plots
COLORS = {
    "CDI":            "#4472C4",
    "DI":             "#ED7D31",
    "SI":             "#70AD47",
    "TBS":            "#FFC000",
    "GCA_sequential": "#7030A0",
    "GCA_flat_joint": "#00B0F0",
    "GCA_oracle":     "#7030A0",
    "GCA_estimation": "#C00000",
}

POLICY_LABELS = {
    "CDI":            "CDI",
    "DI":             "DI",
    "SI":             "SI",
    "TBS":            "TBS",
    "GCA_sequential": "GCA-DS (Sequential)",
    "GCA_flat_joint": "GCA-DS (Flat-Joint)",
    "GCA_oracle":     "GCA-DS (Oracle)",
    "GCA_estimation": "GCA-DS-E (Estimation)",
}

def set_thesis_style():
    """Apply consistent thesis-ready matplotlib style."""
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "lines.linewidth":    1.5,
        "lines.markersize":   5,
    })

def save_figure(fig, filename):
    """Save figure to output folder as both PDF and PNG."""
    path_pdf = get_output_path(filename + ".pdf")
    path_png = get_output_path(filename + ".png")
    fig.savefig(path_pdf)
    fig.savefig(path_png)
    print(f"Saved: {path_pdf}")
    plt.close(fig)