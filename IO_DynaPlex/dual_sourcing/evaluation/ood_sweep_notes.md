# Out-of-distribution sweep: sequential GCA-DS, wide-adjusted-b range, N=100000

Date: 2026-08-24/25

## Purpose

Empirical test of the theoretical Lipschitz-continuity bound (Methodology Sec.1,
Proposition 1): does GCA-DS's performance degrade gracefully as instances move away
from the training range, the way the theory predicts, or does it break down sharply?

## Methodology

One-parameter-at-a-time sweep around a central baseline instance (mu=7, sigma=3.5, h=1,
b=2, c_e=2.5, c_r=0.5, l_e=2, l_r=4 - all within the wide-adjusted-b training range),
varying one parameter across a grid that crosses the training-range boundary while
holding the others fixed at the baseline. CDI/DI/SI/TBS retuned per instance.
GCA-DS: sequential, N=100000 (the run validated in
`wideadjb_n100000_sequential_statistical_comparison_notes.md`). 300 trajectories x 2000
periods per instance. Spec: `eval_spec_ood_sweep.json`, raw data:
`ood_sweep_wideadjb_n100000_sequential.json`.

## Two design issues hit during this run (documented, not silently fixed)

1. **cost_ratio_ce_over_cr, sweep value 1.0 (c_e=c_r=0.5) failed**: the MDP constructor
   requires `max_c > min_c` strictly; setting c_e equal to c_r violates this. Trivial
   edge-case bug in the sweep's instance generation (should have started the c_e grid
   strictly above c_r), not a finding. Dropped from the results below.
2. **lead_time_gap, l_r > 5 (i.e. beyond the training l_max) all failed with a Queue
   length error - and this is NOT fixable by just raising an eval-time bound.** The
   trained network's input feature vector includes the state's pipeline/queue vector,
   whose length is fixed at training time by `max_lr` (here 5). A network trained with
   `max_lr=5` structurally cannot accept a state from an instance with `l_r=7`, because
   the input dimensionality itself would differ - this is an architectural constraint,
   not a config oversight. **This is itself a real, reportable finding**: mu, b, and the
   cost ratio are continuous *feature values* the network can be evaluated at arbitrarily
   far outside the training distribution (whether or not the result is any good); lead
   time is structurally different; it needs a retrained network with a larger `max_lr` to
   be tested out-of-distribution at all. Worth stating explicitly as a limitation/insight,
   not glossing over.

## Results

### mu (training range [2,12], baseline=7)

| mu | gap vs CDI | in/out of range |
|---|---|---|
| 2 | -0.31% | in |
| 4 | -2.10% | in |
| 6 | -1.84% | in |
| 8 | -1.36% | in |
| 10 | -1.21% | in |
| 12 | -1.22% | in (boundary) |
| 14 | +1.35% | **OOD** |
| 16 | +10.80% | **OOD** |
| 18 | +27.93% | **OOD** |
| 20 | +52.79% | **OOD** |
| 24 | +114.10% | **OOD** (2x training max) |

**Textbook-clean degradation curve.** Inside the training range, gap stays small and
favorable (roughly -0.3% to -2.1%). At the boundary (mu=12) it's still winning. The
instant mu crosses the boundary it flips to a loss and grows in what looks like a
monotonically worsening, accelerating way - qualitatively consistent with the Lipschitz
bound predicting cost divergence proportional to distance from the training set, though
this sweep does not attempt to estimate the actual Lipschitz constant. This is the
cleanest single result in the OOD sweep and probably the best figure candidate for the
paper (gap vs. mu, with a shaded region marking the training range).

### b (training range [0.1,4], baseline=2)

| b | gap vs CDI | in/out of range |
|---|---|---|
| 0.1 | -62.50% | in |
| 1 | -8.74% | in |
| 2 | -1.82% | in (baseline) |
| 3 | -1.83% | in |
| 4 | -2.12% | in (boundary) |
| 5 | -2.17% | **OOD** |
| 6 | -2.02% | **OOD** |
| 8 | -1.04% | **OOD** |
| 10 | +0.52% | **OOD** |

The huge win at b=0.1 matches the earlier finding that low backlog cost is where GCA-DS's
advantage is largest (see `wideadjb_n100000_sequential_statistical_comparison_notes.md`'s
correlation analysis). **Notably, b generalizes far more gracefully OOD than mu did** -
gap stays small and roughly flat through b=10 (2.5x the training max), only just tipping
into a (tiny, +0.52%) loss at the far end. This asymmetry between mu and b's OOD behavior
is itself worth discussing: not every parameter degrades at the same rate once outside
the training range.

### cost ratio c_e/c_r (c_r fixed at 0.5, c_e swept; training range for c_e is [0.1,4])

| c_e/c_r | c_e | gap vs CDI | in/out of range |
|---|---|---|---|
| 2 | 1 | -2.16% | in |
| 4 | 2 | -2.05% | in |
| 6 | 3 | -2.12% | in |
| 8 | 4 | -3.66% | in (boundary) |
| 10 | 5 | -5.41% | **OOD** |
| 12 | 6 | -4.85% | **OOD** |
| 16 | 8 | +6.12% | **OOD** |
| 20 | 10 | +14.44% | **OOD** |

Similar pattern to mu: graceful (even improving) just past the boundary, then a clear
flip to worsening losses further out (c_e=8, 10 - 2x and 2.5x the training max_c=4).

### lead-time gap l_r - l_e (l_e fixed at 2; only in-range points survived, see issue 2 above)

| l_r - l_e | gap vs CDI |
|---|---|
| 1 | -3.03% |
| 2 | -1.82% |
| 3 | -1.33% |

Small, consistent wins across the only lead-time gaps this network can actually be
evaluated on. No OOD data available for this axis without retraining at a larger
`max_lr` (see issue 2).

## Bottom line for the paper

Three of four swept parameters (mu, b, cost ratio) show graceful behavior at and near
the training boundary, consistent with the Lipschitz-continuity theory, with mu and cost
ratio showing a clear, sharp inflection into worsening losses once meaningfully outside
the range, and b staying flat much further out. The lead-time axis surfaces a genuine,
distinct limitation: OOD generalization is only actually possible for continuous-valued
features, not for a variable that changes the state's dimensionality, which needs a
retrained network at a larger structural bound to test at all - worth stating plainly in
Discussion/Limitations rather than treated as a sweep design failure.

## Raw data

`ood_sweep_wideadjb_n100000_sequential.json` (36 instances across 4 sweep axes, 5 policies
each; 5 instances failed for the two reasons documented above and are excluded from the
tables here but retained in the raw file with their error messages).
