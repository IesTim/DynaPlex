#!/usr/bin/env python3
"""
Generic parameter-sweep runner for dual_sourcing_backlog DCL experiments.

Runs dual_sourcing_gca once per (sweep value x seed), then evaluates the
resulting policy against CDI on a fixed set of instances via
dual_sourcing_validate eval_k2_fixed, and appends one CSV row per
(sweep value, seed, instance) with the resulting gap and wall-clock time.

Runs strictly sequentially (one training job at a time) - launching multiple
heavy dual_sourcing_gca/dual_sourcing_validate processes in parallel has
previously driven this machine's load average past 10000+ (each process
spawns a full hardware-concurrency-sized thread pool for its own rollouts/
trajectories, so N processes in parallel means N nested full-width pools).

Example (M sensitivity sweep):
  python3 scripts/run_sweep.py \\
    --base_mdp_config mdp_config_k2_two_instances_mub_capped.json \\
    --base_dcl_config dcl_config_k2_faster_allactions.json \\
    --sweep_file mdp --sweep_key rollout_M --sweep_values 500,250,100,50 \\
    --seeds 3 \\
    --eval_instances "4,2,1,5,1,2,1.0,60,10,4,3.0,1,2,0.0;8,4,1,15,1,2,1.0,60,26,15,3.0,1,2,0.0" \\
    --output results_m_sweep.csv
"""
import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "src/lib/models/models/dual_sourcing_backlog/configs"
# dual_sourcing_gca/dual_sourcing_validate resolve bare config filenames relative to this
# directory (see dual_sourcing_gca.cpp: system.filepath("mdp_config_examples", ...)) - NOT
# CONFIG_DIR above, which is only the source tree copy that gets synced here via the
# DP_copy_model_config_files cmake target. Temp sweep configs must live here and be passed
# as bare filenames (not absolute paths), or the executable embeds the whole path into its
# output run-directory name and crashes trying to create nested directories.
RUNTIME_CONFIG_DIR = REPO_ROOT / "IO_DynaPlex/mdp_config_examples/dual_sourcing_backlog/configs"
RUNS_DIR = REPO_ROOT / "IO_DynaPlex/dual_sourcing/runs"
GCA_BIN = REPO_ROOT / "out/LinRel/bin/dual_sourcing_gca"
VALIDATE_BIN = REPO_ROOT / "out/LinRel/bin/dual_sourcing_validate"


def load_config(name_or_path):
    p = Path(name_or_path)
    if not p.is_absolute():
        p = CONFIG_DIR / name_or_path
    with open(p) as f:
        return json.load(f)


def write_temp_config(base_config, overrides, filename):
    cfg = dict(base_config)
    cfg.update(overrides)
    with open(RUNTIME_CONFIG_DIR / filename, "w") as f:
        json.dump(cfg, f, indent=4)


def run_gca(dcl_config_filename, mdp_config_filename):
    """Runs dual_sourcing_gca (bare filenames, resolved under RUNTIME_CONFIG_DIR by the
    executable itself) and returns (elapsed_seconds, run_output_identifier)."""
    start = time.time()
    result = subprocess.run(
        [str(GCA_BIN), dcl_config_filename, mdp_config_filename],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=36000,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"dual_sourcing_gca failed (rc={result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    m = re.search(r"Output path:\s*(\S+)", result.stdout)
    if not m:
        raise RuntimeError(f"Could not find 'Output path:' in gca output:\n{result.stdout[-2000:]}")
    identifier = m.group(1)
    return elapsed, identifier


def find_run_dir(identifier):
    candidates = sorted(RUNS_DIR.glob(f"*{identifier.replace('GCA-DS', '')}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"No run directory found matching identifier fragment: {identifier}")
    return candidates[0].name


def run_eval(run_dir_name, instance_args, action_representation="sequential",
             n_traj=300, n_periods=2000):
    """instance_args: [mu,sigma,h,b,l_e,l_r,c_e,max_order_size,S_r,S_e,inv_cap,struct_l_min,struct_l_max,c_r]"""
    args = [str(VALIDATE_BIN), "eval_k2_fixed", run_dir_name] + [str(a) for a in instance_args] + \
           [action_representation, str(n_traj), str(n_periods)]
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"eval_k2_fixed failed (rc={result.returncode}):\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}")
    m = re.search(r"Gap vs CDI:\s*(-?[\d.]+)%", result.stdout)
    if not m:
        raise RuntimeError(f"Could not parse gap from eval output:\n{result.stdout[-1000:]}")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_mdp_config", required=True)
    ap.add_argument("--base_dcl_config", required=True)
    ap.add_argument("--sweep_file", choices=["mdp", "dcl"], required=True)
    ap.add_argument("--sweep_key", required=True)
    ap.add_argument("--sweep_values", required=True, help="comma-separated values (parsed as int if possible, else float, else bool, else str)")
    ap.add_argument("--seeds", type=int, default=1, help="number of rng_seed values to run per sweep value")
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--eval_instances", required=True,
                     help="semicolon-separated instance arg lists (each comma-separated): mu,sigma,h,b,l_e,l_r,c_e,max_order_size,S_r,S_e,inv_cap,struct_l_min,struct_l_max,c_r")
    ap.add_argument("--eval_n_traj", type=int, default=300)
    ap.add_argument("--eval_n_periods", type=int, default=2000)
    ap.add_argument("--action_representation", default="sequential")
    ap.add_argument("--output", required=True, help="CSV output path (repo-relative or absolute)")
    args = ap.parse_args()

    def parse_val(v):
        v = v.strip()
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v

    sweep_values = [parse_val(v) for v in args.sweep_values.split(",")]
    eval_instances = []
    for inst_str in args.eval_instances.split(";"):
        parts = [p.strip() for p in inst_str.split(",")]
        eval_instances.append(parts)

    base_mdp = load_config(args.base_mdp_config)
    base_dcl = load_config(args.base_dcl_config)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sweep_key", "sweep_value", "seed", "instance_idx", "gap_vs_cdi_pct",
                  "train_elapsed_sec", "run_dir", "timestamp"]
    write_header = not out_path.exists()
    csv_file = open(out_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    tmp_mdp_name = "_sweep_tmp_mdp_config.json"
    tmp_dcl_name = "_sweep_tmp_dcl_config.json"

    total = len(sweep_values) * args.seeds
    done = 0
    for val in sweep_values:
        for seed_offset in range(args.seeds):
            seed = args.seed_start + seed_offset
            done += 1
            print(f"\n=== [{done}/{total}] {args.sweep_key}={val} seed={seed} ===", flush=True)

            mdp_overrides = {"rng_seed": seed} if args.sweep_file == "mdp" else {}
            dcl_overrides = {"rng_seed": seed} if args.sweep_file == "dcl" else {}
            if args.sweep_file == "mdp":
                mdp_overrides[args.sweep_key] = val
            else:
                dcl_overrides[args.sweep_key] = val
            # rng_seed always goes on the dcl_config (that's what DCL::DCL reads)
            dcl_overrides["rng_seed"] = seed

            write_temp_config(base_mdp, mdp_overrides, tmp_mdp_name)
            write_temp_config(base_dcl, dcl_overrides, tmp_dcl_name)

            try:
                elapsed, identifier = run_gca(tmp_dcl_name, tmp_mdp_name)
                run_dir = find_run_dir(identifier)
                print(f"  trained in {elapsed:.1f}s -> {run_dir}", flush=True)
            except Exception as e:
                print(f"  TRAINING FAILED: {e}", flush=True)
                writer.writerow({"sweep_key": args.sweep_key, "sweep_value": val, "seed": seed,
                                  "instance_idx": -1, "gap_vs_cdi_pct": "", "train_elapsed_sec": "",
                                  "run_dir": f"FAILED: {e}", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                csv_file.flush()
                continue

            for idx, inst in enumerate(eval_instances):
                try:
                    gap = run_eval(run_dir, inst, args.action_representation,
                                    args.eval_n_traj, args.eval_n_periods)
                    print(f"  instance {idx}: gap vs CDI = {gap:.3f}%", flush=True)
                except Exception as e:
                    print(f"  instance {idx}: EVAL FAILED: {e}", flush=True)
                    gap = ""
                writer.writerow({
                    "sweep_key": args.sweep_key, "sweep_value": val, "seed": seed,
                    "instance_idx": idx, "gap_vs_cdi_pct": gap,
                    "train_elapsed_sec": f"{elapsed:.1f}", "run_dir": run_dir,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                csv_file.flush()

    csv_file.close()
    print(f"\nDone. Results written to {out_path}")


if __name__ == "__main__":
    main()
