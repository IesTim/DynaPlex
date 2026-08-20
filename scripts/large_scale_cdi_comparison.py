#!/usr/bin/env python3
"""
Generates a large set of random held-out test instances within a given parameter range,
tunes CDI for each (tune_k2_cdi), evaluates a trained network's final policy against each
(eval_k2_fixed), and computes summary statistics (mean/std gap, win rate, paired
Wilcoxon signed-rank test) to give statistically meaningful evidence of whether the
network beats CDI across the range, not just on a handful of instances.

Runs strictly sequentially - see run_sweep.py's docstring for why (process-level
parallelism causes severe CPU oversubscription on this machine; DynaPlex already uses a
full-width internal thread pool per process).

Usage:
  python3 scripts/large_scale_cdi_comparison.py \\
    --run_path sequential_mdp_config_wide_adjusted_b_dcl_config_k2_wideadjb_n20000_20260813_012550 \\
    --n_instances 50 \\
    --mu_range 2,12 --b_range 0.1,4.0 --c_range 0.1,4.0 --l_min 1 --l_max 5 --h 1.0 \\
    --max_order_size 270 --inventory_cap_multiplier 3.0 \\
    --seed 999 \\
    --output IO_DynaPlex/dual_sourcing/evaluation/large_scale_wideadjb_comparison.csv
"""
import argparse
import csv
import random
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATE_BIN = REPO_ROOT / "out/LinRel/bin/dual_sourcing_validate"


def sample_instance(rng, mu_range, b_range, c_range, l_min, l_max, h_range):
    mu = round(rng.uniform(*mu_range), 3)
    sigma = round(mu / 2, 3)
    b = round(rng.uniform(*b_range), 3)
    c_e = round(rng.uniform(*c_range), 3)
    c_r = round(rng.uniform(c_range[0], c_e), 3)
    h = round(rng.uniform(*h_range), 3)
    valid_pairs = [(a, bb) for a in range(l_min, l_max) for bb in range(a + 1, l_max + 1)]
    l_e, l_r = rng.choice(valid_pairs)
    return {"mu": mu, "sigma": sigma, "h": h, "b": b, "c_e": c_e, "c_r": c_r, "l_e": l_e, "l_r": l_r}


def tune_cdi(inst, max_order_size, inventory_cap_multiplier):
    args = [str(VALIDATE_BIN), "tune_k2_cdi", str(inst["mu"]), str(inst["sigma"]), str(inst["h"]),
            str(inst["b"]), str(inst["l_e"]), str(inst["l_r"]), str(inst["c_e"]),
            str(max_order_size), str(inventory_cap_multiplier), str(inst["c_r"])]
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"tune_k2_cdi failed:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")
    m_se = re.search(r'"S_e":(\d+)', result.stdout)
    m_sr = re.search(r'"S_r":(\d+)', result.stdout)
    if not (m_se and m_sr):
        raise RuntimeError(f"Could not parse S_e/S_r from:\n{result.stdout[-500:]}")
    return int(m_se.group(1)), int(m_sr.group(1))


def eval_final(run_path, inst, S_e, S_r, max_order_size, inventory_cap_multiplier,
               struct_l_min, struct_l_max, n_traj, n_periods):
    args = [str(VALIDATE_BIN), "eval_k2_fixed", run_path, str(inst["mu"]), str(inst["sigma"]),
            str(inst["h"]), str(inst["b"]), str(inst["l_e"]), str(inst["l_r"]), str(inst["c_e"]),
            str(max_order_size), str(S_r), str(S_e), str(inventory_cap_multiplier),
            str(struct_l_min), str(struct_l_max), str(inst["c_r"]), "sequential",
            str(n_traj), str(n_periods)]
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        raise RuntimeError(f"eval_k2_fixed failed:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")
    m = re.search(r"Gap vs CDI:\s*(-?[\d.]+)%", result.stdout)
    if not m:
        raise RuntimeError(f"Could not parse gap from:\n{result.stdout[-500:]}")
    return float(m.group(1))


def wilcoxon_signed_rank_p(diffs):
    """Minimal two-sided Wilcoxon signed-rank test p-value (normal approximation),
    no scipy dependency required. diffs = gap values (network beats CDI when negative)."""
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n < 5:
        return None  # too few non-zero samples for a meaningful normal approximation
    abs_diffs = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    rank = 1
    sorted_abs = sorted(abs(diffs[i]) for i in range(n))
    # assign average ranks for ties
    idx_sorted = sorted(range(n), key=lambda i: abs(diffs[i]))
    j = 0
    while j < n:
        k = j
        while k + 1 < n and abs(diffs[idx_sorted[k + 1]]) == abs(diffs[idx_sorted[j]]):
            k += 1
        avg_rank = (j + 1 + k + 1) / 2.0
        for t in range(j, k + 1):
            ranks[idx_sorted[t]] = avg_rank
        j = k + 1
    W_pos = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    W_neg = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    W = min(W_pos, W_neg)
    mean_W = n * (n + 1) / 4.0
    std_W = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    if std_W == 0:
        return None
    z = (W - mean_W) / std_W
    # two-sided p-value from standard normal via erf approximation
    import math
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", required=True)
    ap.add_argument("--n_instances", type=int, default=50)
    ap.add_argument("--mu_range", required=True)
    ap.add_argument("--b_range", required=True)
    ap.add_argument("--c_range", required=True)
    ap.add_argument("--l_min", type=int, required=True)
    ap.add_argument("--l_max", type=int, required=True)
    ap.add_argument("--h", type=float, default=None, help="fixed h (mutually exclusive with --h_range)")
    ap.add_argument("--h_range", type=str, default=None, help="e.g. 0.7,1.3 (mutually exclusive with --h)")
    ap.add_argument("--max_order_size", type=int, required=True)
    ap.add_argument("--inventory_cap_multiplier", type=float, default=3.0)
    ap.add_argument("--struct_l_min", type=int, default=0)
    ap.add_argument("--struct_l_max", type=int, default=0)
    ap.add_argument("--n_traj", type=int, default=300)
    ap.add_argument("--n_periods", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    struct_l_min = args.struct_l_min or args.l_min
    struct_l_max = args.struct_l_max or args.l_max

    mu_range = tuple(float(x) for x in args.mu_range.split(","))
    b_range = tuple(float(x) for x in args.b_range.split(","))
    c_range = tuple(float(x) for x in args.c_range.split(","))
    if args.h_range:
        h_range = tuple(float(x) for x in args.h_range.split(","))
    elif args.h is not None:
        h_range = (args.h, args.h)
    else:
        raise SystemExit("must specify either --h or --h_range")

    rng = random.Random(args.seed)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["instance_idx", "mu", "sigma", "h", "b", "c_e", "c_r", "l_e", "l_r",
                  "S_e", "S_r", "gap_vs_cdi_pct", "timestamp"]
    csv_file = open(out_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    gaps = []
    for i in range(args.n_instances):
        inst = sample_instance(rng, mu_range, b_range, c_range, args.l_min, args.l_max, h_range)
        print(f"\n=== instance {i+1}/{args.n_instances}: mu={inst['mu']} b={inst['b']} "
              f"c_e={inst['c_e']} c_r={inst['c_r']} l=({inst['l_e']},{inst['l_r']}) ===", flush=True)
        try:
            S_e, S_r = tune_cdi(inst, args.max_order_size, args.inventory_cap_multiplier)
            print(f"  CDI tuned: S_e={S_e} S_r={S_r}", flush=True)
            gap = eval_final(args.run_path, inst, S_e, S_r, args.max_order_size,
                              args.inventory_cap_multiplier, struct_l_min, struct_l_max,
                              args.n_traj, args.n_periods)
            print(f"  gap vs CDI: {gap:.3f}%", flush=True)
            gaps.append(gap)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            S_e = S_r = gap = ""
        writer.writerow({"instance_idx": i, **inst, "S_e": S_e, "S_r": S_r,
                          "gap_vs_cdi_pct": gap, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
        csv_file.flush()

    csv_file.close()

    if gaps:
        n = len(gaps)
        mean = sum(gaps) / n
        std = (sum((g - mean) ** 2 for g in gaps) / (n - 1)) ** 0.5 if n > 1 else 0.0
        wins = sum(1 for g in gaps if g < 0)
        p = wilcoxon_signed_rank_p(gaps)
        print(f"\n=== SUMMARY over {n} instances ===")
        print(f"mean gap: {mean:.3f}%  std: {std:.3f}%")
        print(f"win rate (network beats CDI): {wins}/{n} ({100*wins/n:.1f}%)")
        print(f"min: {min(gaps):.3f}%  max: {max(gaps):.3f}%")
        if p is not None:
            print(f"Wilcoxon signed-rank two-sided p-value (H0: median gap = 0): {p:.6f}")
        else:
            print("Wilcoxon signed-rank test: not enough non-zero samples")
    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
