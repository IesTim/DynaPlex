# K-scaling stress test: does sequential decomposition's linear scaling hold up?

Date: 2026-08-25

## Purpose

Not a performance-optimization exercise (per explicit instruction: small N, this is about the
*trend* in degradation, not the best achievable policy). A stress test: sequential
decomposition's rollout cost is O(K(m+1)MH), linear in the number of sources K, in contrast to
flat_joint's O((m+1)^K MH). The question is whether "linear" is actually enough in practice as K
grows, or whether some other factor (state-space growth, compounding sequential sub-decision
approximation, sample complexity of K marginal distributions) causes GCA-DS to degrade before
K=5.

## Infrastructure built for this (not present before today)

The classical heuristics (CDI/DI/SI/TBS), `adaptive_cdi` (the usual DCL initial_policy), and
`dual_sourcing_eval`'s instance-building code were all hardcoded to K=2. The core MDP dynamics
(`mdp.cpp`) were already fully K-generic (needed for the theory in Methodology/Appendix), so no
changes were needed there beyond generalizing `fixed_instance` to accept an array-valued `l`
(backward compatible with the existing K=2-only `l_e`/`l_r` form).

Added: `AdaptiveKSourceCDIPolicy` (`adaptive_cdi_k`) - a transparent, deliberately simple K-generic
greedy heuristic (NOT a claim of optimality; the real CDI's exact 2-source cascade does not
generalize past two sources). Each source k (ordered fastest/priciest at k=0 to slowest/cheapest
at k=K-1, per the model's own dominance ordering) targets `NewsvendorFractile(l_k)/K`, processed
fastest-to-slowest so each source only orders the gap left after faster sources already decided
this period. Used as both the DCL initial_policy for training and the evaluation baseline.
`dual_sourcing_eval` got a new `compare_k` mode, parallel to (not a modification of) the existing
`compare` mode, to avoid any regression risk to the K=2 results already validated in this thesis.

## Methodology

K in {2,3,4,5}, sequential action representation only (flat_joint's K=2 failure already
establishes the qualitative argument for why higher K would only be worse there - not worth the
compute to re-demonstrate). mu in [2,12], b in [0.1,4], c in [0.1,4] per source (independently
drawn, sorted descending), l_min=1, l_max=K+3 (structural bound, scaled with K to keep enough
valid lead-time-tuple variety). max_order_size=150, inventory_cap_multiplier=3.0, rollout_M=100.
**N=5000 (small, deliberately, per instruction - this is about the trend, not best-possible
performance), 5 generations**, same network architecture at every K ([256,128,128,128] hidden
layers). Evaluated against `adaptive_cdi_k` on 15 random within-range instances per K (seed
4242+K), 200 trajectories x 1500 periods each.

## Results

| K | GCA-DS mean cost | adaptive_cdi_k mean cost | gap | win rate | training wall-clock | final argmax agreement |
|---|---|---|---|---|---|---|
| 2 | 21.05 | 32.62 | -33.9% | 15/15 | 44m42s | 100% |
| 3 | 21.72 | 40.30 | -43.4% | 15/15 | 1h06m32s | 100% |
| 4 | 23.11 | 49.48 | -46.8% | 15/15 | 1h30m23s | 100% |
| 5 | 25.56 | 55.37 | -45.5% | 15/15 | 1h57m43s | 100% |

## Interpretation

**No breakdown through K=5.** Three independent signals all say the same thing:
1. **Win rate is 15/15 at every K tested** - GCA-DS never loses to the baseline on any instance,
   at any K.
2. **Argmax agreement stays pinned at 100% at every K's final generation** - the same diagnostic
   that collapsed to near-0% when flat_joint failed catastrophically at K=2 (see
   `flatjoint_action_space_failure_notes.md`) shows no sign of strain here at all, all the way to
   K=5.
3. **Training wall-clock grows linearly, not exponentially**: 44m42s -> 66m32s (+22m) ->
   90m23s (+24m) -> 117m43s (+27m). The increments themselves are nearly constant, i.e. this is
   very close to genuinely linear in K, exactly matching the O(K) theoretical prediction.

**One thing worth flagging honestly rather than glossing over, since this is a stress test and
should report the first hint of strain if there is one:** GCA-DS's *own* absolute cost grows a
little faster at the K=4->K=5 step (23.11 -> 25.56, +10.6%) than at the earlier steps (21.05 ->
21.72 -> 23.11, +3.2%, +6.4%), and the *relative* gap vs. the baseline actually narrows slightly
at K=5 (-46.8% -> -45.5%) rather than continuing to widen. This is a mild, not dramatic, signal -
nowhere near the qualitative collapse seen in the flat_joint failure - but it is the first data
point in this stress test that doesn't cleanly continue the trend of the previous three. Whether
this is the beginning of real strain that would become visible at K=6+, or just noise from a
single small-N run per K (n=1 training run per K, no replication), cannot be determined from this
data alone.

## Bottom line for the paper

The honest headline: **sequential decomposition's linear-in-K scaling held up through K=5 in this
stress test** - performance stayed strong (never worse than the baseline on any of 60 evaluated
instances across all K), training diagnostics stayed healthy (100% argmax agreement throughout,
no sign of the collapse seen in flat_joint), and wall-clock cost grew linearly as the theory
predicts. There is a mild uptick in GCA-DS's own absolute cost at K=5 worth mentioning as a
"watch this" observation for future work, but this stress test did not find the point where linear
scaling stops being enough within the K=2..5 range and the 1-day compute budget available.

## Raw data

`kscaling_K{2,3,4,5}_comparison.json` (15 instances x 2 policies each, full per-trajectory
costs). Training logs (per-generation argmax agreement, cost improvement, wall-clock) are in the
run console output only, not yet extracted to a structured JSON - could be added if the paper
wants an exact per-generation diagnostic table.
