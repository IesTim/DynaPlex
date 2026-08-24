# flat_joint action-space infeasibility: evidence for the paper

Date: 2026-08-24

## Purpose

Academic claim to support: flat_joint does not scale to this action space (K=2,
max_order_size=270 => 271*271=73441 joint actions) because the action space overwhelms it either
way you handle it - cap the rollout candidates and it can't find good actions; don't cap them and
training becomes computationally infeasible. This note documents both halves of the evidence.

Range: same wide-adjusted-b range as the sequential N-scaling results (mu[2,12], b[0.1,4],
c[0.1,4], l[1,5], h=1.0, max_order_size=270).

## Evidence 1: capped run performs catastrophically, not just "worse"

Run: `flat_joint_mdp_config_wide_adjusted_b_flatjoint_dcl_config_k2_wideadjb_flatjoint_n20000_20260822_074005`
(N=20000, `SimulateOnlyPromisingActions=true`, `Num_Promising_Actions=32` i.e. a candidate window
of 32 out of 73441 possible actions - about 0.04% coverage). Stopped after generation 3 rather
than running all 5: by generation 2 the failure mode was already unambiguous and each additional
generation cost another ~15-20h of wall clock the project's timeline could not absorb without
changing the conclusion (see timing below).

Training diagnostics (from the training log, not yet evaluation):
- Gen 1: fast (3 min sample generation, heuristic-driven), but training already ended at 0%
  argmax agreement (masked-vs-unmasked argmax, see policytrainer.cpp) and negative cost
  improvement.
- Gen 2: training ended at 0% argmax agreement essentially throughout, cost improvement around
  -45 (i.e. the rollout-assessed cost got ~45x worse, not better, over training).
- Gen 3: argmax agreement recovered somewhat mid-training (peaked ~38%) before collapsing back to
  ~5-7% by the end; cost improvement ended around -15.3 (still catastrophic, if less so than gen
  2).

Evaluation of the generation-3 checkpoint against tuned CDI/DI/SI/TBS on 50 random instances
(seed 2024, `eval_spec_flatjoint_failure.json`, `dual_sourcing_eval compare`, 300 traj x 2000
periods/instance):

| Metric | Value |
|---|---|
| win rate vs CDI | **0/50 (0.0%)** |
| mean gap vs CDI | **+274.6%** |
| min (best case) | +107.3% |
| max (worst case) | +608.3% |

Every single instance is a loss, and not a marginal one - on average the capped flat_joint policy
costs nearly 3.75x what tuned CDI costs. Raw data: `flatjoint_failure_wideadjb_n20000_gen3_comparison.json`.

## Evidence 2: computational cost explodes once the network drives sampling

Per-phase wall-clock timing from the same run's log:

| Phase | Time |
|---|---|
| Gen 1 sample generation (heuristic-driven, `adaptive_cdi`) | 3 min |
| Gen 1 training | 56 min |
| Gen 2 sample generation (NN-driven rollouts) | **19h 6min** |
| Gen 2 training | 1h 38min |
| Gen 3 sample generation | ~15-17h (estimated from partial progress before stopping) |

Once the trained network (not the cheap heuristic) drives DCL's rollout-based sample generation,
wall-clock cost exploded by roughly 400x per generation (3 min -> ~19h) at unchanged N=20000.
The mechanism: every rollout-comparison step requires a forward pass through a network with a
73441-wide output layer (one score per joint action), and this happens M x H x candidates times
per training sample. Even with only 32 candidates being *compared*, the network itself is
expensive to *evaluate* because of its output dimension.

For comparison: the entire sequential N=100000 run on the same range (5x more data, uncapped
action enumeration, but only 542 actions rather than 73441) took **33.9 hours total** - less
than this single capped flat_joint run took to get through 3 of 5 generations at N=20000 (5x
less data). This wall-clock comparison is itself a strong, citable number.

**On the "uncapped is infeasible" claim specifically:** a dedicated tiny-N timing probe
(`dcl_config_k2_wideadjb_flatjoint_uncapped_probe.json`, N=500, full action enumeration, no
capping) was prepared but not yet run at time of writing - queued as follow-up if a clean
extrapolated infeasibility number is wanted in addition to the capped-run's own dramatic slowdown
(gen 2 alone taking 19h at N=20000 with only 32 candidates strongly suggests the uncapped case,
evaluating all 73441 candidates instead of 32, would be on the order of ~2300x slower still per
rollout-comparison call, i.e. entirely impractical - but this has not been directly measured yet).

## Bottom line for the paper

Two independent, converging lines of evidence: (1) capped flat_joint's performance collapses
completely (0% win rate, +274.6% mean cost, catastrophic 0% argmax agreement during training) -
the candidate window cannot meaningfully cover a 73441-action space; (2) even capped, the
per-generation wall-clock cost explodes ~400x once the network (rather than a cheap heuristic)
must be evaluated to drive sampling, an effect that would only be worse without capping. Sequential's
action space (2*(max_order_size+1)=542 here) sidesteps both problems, which is the structural
reason to expect it to keep outperforming flat_joint as K (and hence the joint action space)
grows further - directly motivating the K-scaling experiment (test 6).

## Raw data

- `flatjoint_failure_wideadjb_n20000_gen3_comparison.json` (50 instances x 5 policies)
- Training log timing figures are from the run's console output (not yet saved as a separate
  structured JSON - could be added if the paper wants exact per-epoch timing series).
