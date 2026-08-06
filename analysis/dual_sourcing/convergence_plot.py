"""
Produces convergence plot showing GCA-DS cost vs CDI across training generations.
Compares sequential and flat-joint action representations.
Output: output/convergence.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_convergence_results,
    compute_gap,
    set_thesis_style,
    save_figure,
    COLORS,
)

def plot_convergence(ax, data, action_repr, instance_name=None):
    """Plot cost gap vs CDI across generations for one action representation."""

    generations = data["generations"]
    gen_numbers = []
    gaps = []

    for gen in generations:
        instances = gen["instances"]
        # Filter by instance if specified
        if instance_name:
            instances = [i for i in instances if i["instance"] == instance_name]

        # Average gap across instances
        gen_gaps = []
        for inst in instances:
            gca_cost = inst.get("GCA_cost")
            cdi_cost = inst.get("CDI_cost")
            if gca_cost and cdi_cost:
                gen_gaps.append(compute_gap(gca_cost, cdi_cost))

        if gen_gaps:
            gen_numbers.append(gen["generation"])
            gaps.append(np.mean(gen_gaps))

    label = "Sequential" if action_repr == "sequential" else "Flat-Joint"
    color = COLORS["GCA_sequential"] if action_repr == "sequential" else COLORS["GCA_flat_joint"]

    ax.plot(gen_numbers, gaps, marker="o", label=label, color=color)

def main():
    set_thesis_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Load results for both representations
    for action_repr in ["sequential", "flat_joint"]:
        try:
            data = load_convergence_results(action_repr)
            plot_convergence(ax, data, action_repr)
        except FileNotFoundError:
            print(f"No convergence results for {action_repr} — skipping.")

    # Reference line at 0 (CDI performance)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0,
               label="CDI (reference)")

    ax.set_xlabel("Training Generation")
    ax.set_ylabel("Average Gap vs. CDI (\\%)")
    ax.set_title("GCA-DS Convergence Across Training Generations")
    ax.legend()

    save_figure(fig, "convergence")

if __name__ == "__main__":
    main()
