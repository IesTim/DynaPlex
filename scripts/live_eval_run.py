#!/usr/bin/env python3
"""
Launches a dual_sourcing_gca training run in the background, then polls for each
generation's checkpoint (dcl_policy_gen<N>) to appear on disk and immediately
evaluates it against a fixed validation set (small_range_validation_set.json) via
the eval_k2_gen debug command - giving a live, per-generation trend of gap-vs-CDI
while training is still in progress, instead of only a single final number.

Per-generation checkpoints are saved under a path keyed by the training MDP's
Identifier() (see PolicyTrainer::PathToPolicy in policytrainer.cpp), which this
script recovers from the run's own run_info.json once training has started.

Usage:
  python3 scripts/live_eval_run.py \\
    --mdp_config mdp_config_k2_small_range.json \\
    --dcl_config dcl_config_k2_range_verysmall.json \\
    --validation_set IO_DynaPlex/dual_sourcing/evaluation/small_range_validation_set.json \\
    --run_label very_small \\
    --output IO_DynaPlex/dual_sourcing/evaluation/live_eval_verysmall.csv \\
    --eval_n_traj 100 --eval_n_periods 1000
"""
import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CONFIG_DIR = REPO_ROOT / "IO_DynaPlex/mdp_config_examples/dual_sourcing_backlog/configs"
RUNS_DIR = REPO_ROOT / "IO_DynaPlex/dual_sourcing/runs"
IO_ROOT = REPO_ROOT / "IO_DynaPlex"
GCA_BIN = REPO_ROOT / "out/LinRel/bin/dual_sourcing_gca"
VALIDATE_BIN = REPO_ROOT / "out/LinRel/bin/dual_sourcing_validate"


def find_run_dir_and_identifier(mdp_config_name, dcl_config_name, started_after):
    """Poll for the run directory dual_sourcing_gca just created, and once run_info.json
    exists, read the mdp_identifier out of it."""
    mdp_stem = Path(mdp_config_name).stem
    dcl_stem = Path(dcl_config_name).stem
    while True:
        candidates = [p for p in RUNS_DIR.glob(f"*{mdp_stem}_{dcl_stem}_*")
                      if p.is_dir() and p.stat().st_mtime >= started_after]
        if candidates:
            run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
            info_path = run_dir / "run_info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text())
                    if "mdp_identifier" in info:
                        return run_dir.name, info["mdp_identifier"]
                except json.JSONDecodeError:
                    pass
        time.sleep(2)


def checkpoint_path(mdp_identifier, generation):
    return IO_ROOT / mdp_identifier / f"dcl_policy_gen{generation}.json"


def run_eval_gen(run_dir_name, generation, inst, args):
    c_e = inst.get("c_e", args.c_e)
    cli_args = [str(VALIDATE_BIN), "eval_k2_gen", run_dir_name, str(generation),
                str(inst["mu"]), str(inst["sigma"]), str(inst["h"]), str(inst["b"]),
                str(inst["l_e"]), str(inst["l_r"]), str(c_e),
                str(args.max_order_size), str(inst["S_r"]), str(inst["S_e"]),
                str(args.inventory_cap_multiplier), str(args.struct_l_min), str(args.struct_l_max),
                str(inst["c_r"]), args.action_representation,
                str(args.eval_n_traj), str(args.eval_n_periods)]
    result = subprocess.run(cli_args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"eval_k2_gen failed:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")
    m = re.search(r"Gap vs CDI:\s*(-?[\d.]+)%", result.stdout)
    if not m:
        raise RuntimeError(f"Could not parse gap from:\n{result.stdout[-500:]}")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mdp_config", required=True)
    ap.add_argument("--dcl_config", required=True)
    ap.add_argument("--validation_set", required=True)
    ap.add_argument("--run_label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval_n_traj", type=int, default=100)
    ap.add_argument("--eval_n_periods", type=int, default=1000)
    ap.add_argument("--action_representation", default="sequential")
    args = ap.parse_args()

    val_set = json.loads(Path(args.validation_set).read_text())
    args.c_e = val_set.get("c_e")  # fallback shared value; per-instance "c_e" key takes precedence if present
    args.max_order_size = val_set["max_order_size"]
    args.inventory_cap_multiplier = val_set["inventory_cap_multiplier"]
    args.struct_l_min = val_set["struct_l_min"]
    args.struct_l_max = val_set["struct_l_max"]
    instances = val_set["instances"]

    with open(RUNTIME_CONFIG_DIR / args.dcl_config) as f:
        dcl_config = json.load(f)
    num_gens = dcl_config["num_gens"]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_label", "generation", "instance_idx", "gap_vs_cdi_pct", "timestamp"]
    write_header = not out_path.exists()
    csv_file = open(out_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    print(f"[{args.run_label}] launching training: {args.dcl_config} + {args.mdp_config}", flush=True)
    start_time = time.time()
    proc = subprocess.Popen([str(GCA_BIN), args.dcl_config, args.mdp_config], cwd=REPO_ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    run_dir_name, mdp_identifier = find_run_dir_and_identifier(args.mdp_config, args.dcl_config, start_time)
    print(f"[{args.run_label}] run_dir={run_dir_name} mdp_identifier={mdp_identifier}", flush=True)

    # Checkpoints are saved under a path keyed only by mdp_identifier (derived from mdp_config
    # content), NOT by run/dcl_config - so any earlier run sharing the same mdp_config left its
    # own dcl_policy_gen*.json/.pth files sitting in this same directory. Without clearing these,
    # the "does the checkpoint file exist yet" poll below would immediately find - and evaluate -
    # a stale leftover from a *previous* run instead of waiting for this run's own fresh one.
    identifier_dir = IO_ROOT / mdp_identifier
    if identifier_dir.exists():
        stale = list(identifier_dir.glob("dcl_policy_gen*.json")) + list(identifier_dir.glob("dcl_policy_gen*.pth"))
        for f in stale:
            f.unlink()
        if stale:
            print(f"[{args.run_label}] cleared {len(stale)} stale checkpoint file(s) from a previous "
                  f"run sharing this mdp_config before starting the live-eval poll loop", flush=True)

    evaluated_gens = set()
    while True:
        retcode = proc.poll()
        for gen in range(1, num_gens + 1):
            if gen in evaluated_gens:
                continue
            if checkpoint_path(mdp_identifier, gen).exists():
                # Give the .pth weights file a moment to finish writing alongside the .json.
                time.sleep(3)
                print(f"[{args.run_label}] generation {gen} checkpoint ready, evaluating "
                      f"{len(instances)} validation instances...", flush=True)
                gaps = []
                for inst in instances:
                    try:
                        gap = run_eval_gen(run_dir_name, gen, inst, args)
                    except Exception as e:
                        print(f"[{args.run_label}] gen {gen} instance {inst['idx']}: EVAL FAILED: {e}", flush=True)
                        gap = None
                    if gap is not None:
                        gaps.append(gap)
                    writer.writerow({"run_label": args.run_label, "generation": gen,
                                      "instance_idx": inst["idx"],
                                      "gap_vs_cdi_pct": gap if gap is not None else "",
                                      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                    csv_file.flush()
                if gaps:
                    mean_gap = sum(gaps) / len(gaps)
                    print(f"[{args.run_label}] GEN {gen} SUMMARY: mean gap vs CDI across "
                          f"{len(gaps)} instances = {mean_gap:.2f}%", flush=True)
                evaluated_gens.add(gen)
        if retcode is not None and len(evaluated_gens) >= num_gens:
            break
        if retcode is not None and len(evaluated_gens) < num_gens:
            # Process ended but not all generations were seen (e.g. crashed) - one more
            # sweep in case the last checkpoint just appeared, then give up.
            time.sleep(3)
            still_missing = [g for g in range(1, num_gens + 1) if g not in evaluated_gens]
            if still_missing and not any(checkpoint_path(mdp_identifier, g).exists() for g in still_missing):
                remaining_stdout = proc.stdout.read() if proc.stdout else ""
                print(f"[{args.run_label}] training process ended (rc={retcode}) with generations "
                      f"{still_missing} never producing a checkpoint - stopping.\n"
                      f"--- process output tail ---\n{remaining_stdout[-2000:]}", flush=True)
                break
            continue
        time.sleep(5)

    stdout, _ = proc.communicate()
    elapsed = time.time() - start_time
    print(f"[{args.run_label}] training process finished (rc={proc.returncode}) after {elapsed:.1f}s total", flush=True)
    csv_file.close()
    print(f"[{args.run_label}] Done. Results in {out_path}", flush=True)


if __name__ == "__main__":
    main()
