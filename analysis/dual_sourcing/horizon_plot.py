"""
Produces horizon plot showing GCA-DS-E converging to GCA-DS-Oracle
as estimation improves over time.
Output: output/horizon.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_horizon_results,
    set_thesis_style,
    save_figure,
    COLORS,
    POLICY_LABELS,
)

def compute_running_average(costs):
    """Compute running average of per-period costs."""
    cumsum = np.cumsum(costs)
    counts = np.arange(1, len(costs) + 1)
    return cumsum / counts

def main():
    set_thesis_style()

    try:
        data = load_horizon_results()
    except FileNotFoundError:
        print("No horizon results found.")
        return

    periods = data["periods"]
    T = len(periods)
    period_numbers = list(range(1, T + 1))

    oracle_costs = [p["GCA_oracle_cost"] for p in periods]
    est_costs    = [p["GCA_estimation_cost"] for p in periods]

    oracle_avg = compute_running_average(oracle_costs)
    est_avg    = compute_running_average(est_costs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1 — per-period cost
    ax = axes[0]
    ax.plot(period_numbers, oracle_costs, alpha=0.5,
            color=COLORS["GCA_oracle"],
            label=POLICY_LABELS["GCA_oracle"])
    ax.plot(period_numbers, est_costs, alpha=0.5,
            color=COLORS["GCA_estimation"],
            label=POLICY_LABELS["GCA_estimation"])
    ax.set_xlabel("Period")
    ax.set_ylabel("Cost per Period")
    ax.set_title("Per-Period Cost over Time")
    ax.legend()

    # Plot 2 — cumulative average cost
    ax = axes[1]
    ax.plot(period_numbers, oracle_avg,
            color=COLORS["GCA_oracle"],
            label=POLICY_LABELS["GCA_oracle"])
    ax.plot(period_numbers, est_avg,
            color=COLORS["GCA_estimation"],
            label=POLICY_LABELS["GCA_estimation"])
    ax.set_xlabel("Period")
    ax.set_ylabel("Cumulative Average Cost")
    ax.set_title("Convergence of Estimation Policy to Oracle")
    ax.legend()

    fig.suptitle("Online Estimation: GCA-DS-E vs GCA-DS-Oracle",
                 fontsize=13)
    plt.tight_layout()

    save_figure(fig, "horizon")

if __name__ == "__main__":
    main()