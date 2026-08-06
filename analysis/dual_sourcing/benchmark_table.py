"""
Produces LaTeX table comparing all policies across all 18 benchmark instances.
Output: output/benchmark_table.tex
"""

import numpy as np
from utils import (
    load_benchmark_results,
    compute_gap,
    get_output_path,
)

# Policies to include in table
POLICIES = ["CDI", "DI", "SI", "TBS", "GCA_sequential", "GCA_flat_joint"]

POLICY_HEADERS = {
    "CDI":            "CDI",
    "DI":             "DI",
    "SI":             "SI",
    "TBS":            "TBS",
    "GCA_sequential": r"GCA-DS$_\text{seq}$",
    "GCA_flat_joint": r"GCA-DS$_\text{fj}$",
}

def format_cost(cost):
    """Format cost to 2 decimal places."""
    if cost is None:
        return "--"
    return f"{cost:.2f}"

def format_gap(gap):
    """Format gap with sign and 1 decimal place."""
    if gap is None:
        return "--"
    return f"{gap:+.1f}\\%"

def instance_label(inst):
    """Create a readable instance label."""
    l_r = inst["l_r"]
    mu  = inst["mu"]
    b   = inst["b"]
    c_e = inst["c_e"]
    return f"$l_r={l_r}, \\mu={int(mu)}, b={int(b)}, c_e={c_e}$"

def build_table(data):
    instances = data["instances"]

    lines = []

    # Table header
    n_policies = len(POLICIES)
    col_spec = "l" + "rr" * n_policies
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Benchmark results: cost and gap vs.\ CDI for all policies across 18 instances. "
                 r"Negative gap indicates improvement over CDI.}")
    lines.append(r"\label{tab:benchmark}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Policy header row
    header = "Instance"
    for p in POLICIES:
        header += f" & \\multicolumn{{2}}{{c}}{{{POLICY_HEADERS[p]}}}"
    lines.append(header + r" \\")

    # Cost/Gap subheader
    subheader = ""
    for p in POLICIES:
        subheader += " & Cost & Gap"
    lines.append(r"\cmidrule(lr){2-" + str(1 + 2 * n_policies) + "}")
    lines.append("Instance" + subheader + r" \\")
    lines.append(r"\midrule")

    # Group instances by mu for visual separation
    prev_mu = None
    for inst in instances:
        mu = inst["mu"]
        if prev_mu is not None and mu != prev_mu:
            lines.append(r"\midrule")
        prev_mu = mu

        row = instance_label(inst)

        cdi_cost = inst.get("CDI", {}).get("cost")

        for p in POLICIES:
            policy_data = inst.get(p)
            if policy_data is None:
                row += " & -- & --"
            else:
                cost = policy_data.get("cost")
                gap  = compute_gap(cost, cdi_cost) if cost and cdi_cost and p != "CDI" else None
                row += f" & {format_cost(cost)}"
                if p == "CDI":
                    row += " & --"
                else:
                    row += f" & {format_gap(gap)}"

        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def main():
    data = load_benchmark_results()
    table = build_table(data)

    out_path = get_output_path("benchmark_table.tex")
    with open(out_path, "w") as f:
        f.write(table)
    print(f"Saved: {out_path}")
    print("\nPreview:")
    print(table)

if __name__ == "__main__":
    main()