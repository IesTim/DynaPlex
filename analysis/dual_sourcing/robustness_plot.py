"""
Produces robustness plots showing GCA-DS performance as parameters
vary outside the training distribution.
Output: output/robustness_<parameter>.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_parameter_results,
    compute_gap,
    set_thesis_style,
    save_figure,
    COLORS,
    POLICY_LABELS,
)

PARAMETERS = {
    "mu":  ("Demand Mean $\\mu$",       True),
    "b":   ("Backlog Penalty $b$",       True),
    "c_e": ("Expedited Cost $c_e$",      True),
}

def plot_robustness(ax, data, parameter_label, has_training_bounds):
    """Plot gap vs CDI as a parameter varies."""

    points = data["parameter_points"]
    param_values = [p["value"] for p in points]

    # Plot GCA gap
    gca_gaps = []
    for p in points:
        gca_cost = p.get("GCA_flat_cost")
        cdi_cost = p.get("CDI_cost")
        if gca_cost and cdi_cost:
            gca_gaps.append(compute_gap(gca_cost, cdi_cost))
        else:
            gca_gaps.append(None)

    valid = [(x, y) for x, y in zip(param_values, gca_gaps) if y is not None]
    if valid:
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker="o",
                color=COLORS["GCA_flat_joint"],
                label="GCA-DS")

    # Training distribution bounds
    if has_training_bounds:
        train_min = data.get("train_min")
        train_max = data.get("train_max")
        if train_min is not None and train_max is not None:
            ax.axvspan(train_min, train_max, alpha=0.1,
                      color="gray", label="Training range")
            ax.axvline(train_min, color="gray",
                      linestyle="--", linewidth=1.0)
            ax.axvline(train_max, color="gray",
                      linestyle="--", linewidth=1.0)

    # Reference line at 0
    ax.axhline(0, color="black", linestyle="--",
              linewidth=1.0, label="CDI (reference)")

    ax.set_xlabel(parameter_label)
    ax.set_ylabel("Gap vs. CDI (\\%)")
    ax.legend()

def main():
    set_thesis_style()

    for param, (label, has_bounds) in PARAMETERS.items():
        try:
            data = load_parameter_results(param)
        except FileNotFoundError:
            print(f"No results for parameter {param} — skipping.")
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_robustness(ax, data, label, has_bounds)
        ax.set_title(f"Robustness: GCA-DS Performance vs. {label}")
        save_figure(fig, f"robustness_{param}")

if __name__ == "__main__":
    main()