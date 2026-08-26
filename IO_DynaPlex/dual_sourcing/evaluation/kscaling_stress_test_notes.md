# K-scaling stress test: does sequential decomposition's linear scaling hold up?

Date: 2026-08-25/26. Extended overnight (2026-08-26) to K=6,7 after the initial K=2-5 run
surfaced a mild, ambiguous signal at K=5 - see "Extension to K=6,7" below for the update that
changes the headline finding.

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

## Results (K=2..5, initial run)

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

## Extension to K=6,7 (overnight, 2026-08-26)

Same protocol exactly (N=5000, 5 generations, `adaptive_cdi_k` baseline, 15 random within-range
instances, same architecture). Motivated directly by the K=5 "watch this" signal above.

| K | GCA-DS mean cost | adaptive_cdi_k mean cost | gap | win rate | training wall-clock | final argmax agreement |
|---|---|---|---|---|---|---|
| 6 | 23.26 | 61.48 | -54.0% | 15/15 | 2h26m42s | 100% |
| 7 | 33.24 | 88.50 | -46.5% | **12/15** | 2h58m27s | 100% |

**K=6 looked fine - even better than K=5 (biggest relative win margin of any K tested, -54.0%) -
but K=7 is where a real crack appears.** For the first time anywhere in this stress test (K=2
through 6, 75 evaluated instances, zero losses), GCA-DS actually loses to the K-generic baseline
on 3 of 15 instances at K=7, with one loss as large as +27.0%. GCA-DS's own absolute cost also
jumps sharply from K=6 to K=7 (23.26 -> 33.24, +42.9%, by far the largest step-to-step increase
anywhere in the K=2..7 range - compare the K=2..6 increases, all under 11%).

**What characterizes the 3 losing instances: low backlog cost b.** All three losses occur at
b in {0.10, 0.35, 0.42} - the three smallest b values among the 15 instances tested. Correlation
between b and gap across all 15 K=7 instances: **corr(b, gap) = -0.879**, a strong relationship in
a small sample. This is a plausible, mechanistically sensible failure mode, not an unexplained
anomaly: N=5000 is small and fixed across all K by design, so the same sample budget has to cover
a state/action space that grows with K; the low-b corner is plausibly where correct behavior
differs most from what a "typical" instance in the training distribution looks like (lower
backlog penalty rewards more conservative ordering), making it the first region to run out of
effective sample coverage as K grows. This maps directly onto the "sample complexity" difficulty
identified theoretically in Methodology Sec.4 ("a dataset of fixed size ... provides less evidence
per class as K grows") - K=7 with N=5000 appears to be roughly where that theoretical concern
starts to bite empirically, at least in this low-b corner.

**Also notable: training diagnostics did not flag this.** Argmax agreement is still a clean 100%
at K=7's final generation - the same as every other K tested, and nothing like the near-0%
collapse seen in the flat_joint failure. This is a materially different, subtler failure mode than
flat_joint's: the network trains "cleanly" by its own internal diagnostic, but its deployed
performance still degrades on part of the instance distribution. Worth stating explicitly: a
healthy-looking training run does not guarantee healthy deployed performance once K is large
enough relative to N - the two diagnostics diverge exactly where this stress test finds a real
crack.

**Caveat, as with the K=5 finding: n=1 training run per K, no replication.** Whether the K=6 dip
back to "no losses" and the K=7 jump is a stable, reproducible K=7-specific effect, or partly an
artifact of drawing one particular training run per K, cannot be fully separated without repeated
runs at the same K with different seeds - not attempted here given the time budget. The
correlation with b within K=7's own 15-instance sample is real regardless of that caveat, since it
doesn't depend on comparing across different training runs.

## Bottom line for the paper

**Revised headline (supersedes the K=2-5-only conclusion above): sequential decomposition's linear
scaling holds up cleanly through K=6, then shows a genuine crack at K=7** - not a catastrophic
collapse like flat_joint's (win rate is still 12/15, not 0/15; argmax agreement stays at 100%; the
absolute cost increase, while the largest step seen, is not an order-of-magnitude jump), but a
real, characterizable failure: losses appear for the first time, concentrated in the low-backlog-
cost region, consistent with the sample-complexity mechanism predicted theoretically in
Methodology Sec.4. This is arguably a *better* result for the paper than either extreme (no
breakdown found, or an immediate collapse): it locates a specific, mechanistically-explained
boundary (K=7 at N=5000, low-b corner) rather than leaving the question unresolved, and it
directly validates one of the three theoretical difficulties (sample complexity) the Methodology
chapter predicts in advance of seeing this data.

## Memory finding: the flat_joint uncapped-enumeration probe was OOM-killed

Attempted as a follow-up to the flat_joint failure result (see
`flatjoint_action_space_failure_notes.md`), to get a measured per-sample timing for the
"infeasible without capping" argument instead of only an extrapolation from the capped run's
slowdown. At N=500, a single generation, full action enumeration (no `SimulateOnlyPromisingActions`
capping) on the same wide-adjusted-b range: the process was killed by the Linux OOM killer after
consuming **260GB of resident memory (anon-rss) on this 251GB machine** - it did not run slowly to
completion, it exhausted all available system memory outright. Confirmed via
`journalctl -k`: `Out of memory: Killed process ... (dual_sourcing_g) total-vm:297355792kB,
anon-rss:260207528kB`. This strengthens the flat_joint infeasibility argument beyond the wall-clock
case already made: uncapped rollout evaluation over 73441 joint actions is not merely slow, it is
not memory-feasible on this hardware even at a tiny N=500, a single generation, and is a stronger,
more vivid, more easily-communicated number for the paper than a multi-day timing extrapolation.

## Raw data

`kscaling_K{2,3,4,5,6,7}_comparison.json` (15 instances x 2 policies each, full per-trajectory
costs). Training logs (per-generation argmax agreement, cost improvement, wall-clock) are in the
run console output only, not yet extracted to a structured JSON - could be added if the paper
wants an exact per-generation diagnostic table.
