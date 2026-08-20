# Large-scale statistical CDI comparison: wide-adjusted-b range

Date: 2026-08-13

## Methodology

Evaluated the final trained policy from `sequential_mdp_config_wide_adjusted_b_dcl_config_k2_wideadjb_n20000_20260813_012550`
(N=20000, 5 generations, gen5/final checkpoint) against **50 freshly-sampled random instances**
drawn uniformly from the training range (mu in [2,12], b in [0.1,4], c_e/c_r in [0.1,4] with
c_r <= c_e, l pairs uniform over valid (l_e,l_r) with 1<=l_e<l_r<=5, h=1.0 fixed), using a
different RNG seed than the 10-instance curated validation set used during training monitoring.
CDI tuned per-instance via the same coordinate-ascent line search used throughout this project.
Evaluation: `eval_k2_fixed`, 300 trajectories x 2000 periods per instance.

This is a materially different (and more rigorous) test than the earlier 10-instance validation
set: that set was deliberately stratified (corners + center + a modest random sample) and used
throughout training for live monitoring, so results on it are not a clean held-out estimate of
performance on "the range" as a whole. This 50-instance set is a proper random sample, closer to
what a reviewer would expect as evidence for a population-level claim.

## Result: the network does NOT reliably beat CDI on this range

| Metric | Value |
|---|---|
| n | 50 |
| mean gap vs CDI | **+7.69%** (positive = network loses) |
| std | 17.17% |
| win rate (network beats CDI) | **16/50 (32.0%)** |
| min (best win) | -29.43% |
| max (worst loss) | +86.32% |
| Wilcoxon signed-rank p-value (H0: median gap = 0) | **0.001** |

**This is a statistically significant result in the wrong direction:** the network loses to CDI
on average, and the Wilcoxon test rejects the null hypothesis of no difference (p=0.001) in
favor of CDI, not the network. This directly contradicts the earlier finding (all 5 generations
of the same run beating CDI on the 10-instance curated set, overall mean -6.77%) — that earlier
result does not generalize to a proper random sample of the training range.

**Implication:** the 10-instance curated validation set materially overstated performance on
this range. For the paper, any "beats CDI" claim on the wide-adjusted-b range needs to be
re-evaluated against a properly random test set like this one, not the smaller curated set used
for live per-generation monitoring during training.

## What correlates with losing?

Pearson correlation of gap-vs-CDI with instance parameters (n=50):

| Parameter | correlation with gap |
|---|---|
| b (backlog cost) | **+0.47** |
| \|c_e - c_r\| (cost-difference between sources) | +0.18 (+0.11 excluding the outlier) |
| mu (demand rate) | -0.16 |

**b has the strongest relationship, and it's moderate, not weak.** Higher backlog cost
correlates with worse (more losing) performance for the network relative to CDI - even within
the *already-adjusted* [0, 4] range. The originally-suspected "close source costs" pattern
(noticed while monitoring individual instances live) turns out to be weak and driven mostly by
one outlier instance (mu=6.68, b=2.21, c_e=1.08, c_r=0.91, gap=+86.32% - by far the largest
single loss in the sample); with that instance excluded the |c_e-c_r| correlation nearly
vanishes (0.11).

**Implication for the "increase the backlog range further" question:** this correlation is a
warning sign, not a green light. Performance already appears to degrade as b increases within
the current [0,4] range; further widening the b range should be expected to make the average
result worse, not better, unless something else changes (e.g. more training data specifically
weighted toward higher-b instances, or a larger action space to better represent the higher
base-stock targets that come with higher b).

## Next step

Proceeding with a b-range extension test (b in [0,20], same N=20000, same other settings) to
directly test whether this degrades further as the correlation predicts, or whether the small
N=20000 sample budget (rather than b itself) is the actual limiting factor. If the b in [0,20]
run also underperforms substantially, this points to needing a proper large-scale statistical
comparison (like this one) built for that range too, rather than relying on curated validation
sets going forward - and likely needing to scale N further before drawing conclusions about b's
true effect on generalization.

## Raw data

`large_scale_wideadjb_comparison.csv` (50 rows, per-instance parameters, tuned CDI S_e/S_r, and
gap).
