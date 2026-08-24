# Sequential GCA-DS, wide-adjusted-b range, N=100000: statistical comparison + loss analysis

Date: 2026-08-24

## Context

Range: mu in [2,12], b in [0.1,4], c_e/c_r both in [0.1,4], l_min=1/l_max=5, h=1.0 fixed,
max_order_size=270, inventory_cap_multiplier=3.0, rollout_M=100. This is the range locked in
for the paper's core N-scaling experiment (test 1) and the algorithm used for downstream tests
2-4/7, per the user's explicit choice over the small range.

Run: `sequential_mdp_config_wide_adjusted_b_dcl_config_k2_wideadjb_n100000_20260820_214806`
(git commit `f798b5c`, 5 generations, 33.9h wall clock - notably faster than the ~52h
extrapolation from the N=20000->50000 scaling trend). Evaluated via the new consolidated
`dual_sourcing_eval compare` pipeline: 100 randomly sampled instances (seed 2024, same seed used
for the N=20000/50000 comparisons on this range for direct comparability), CDI/DI/SI/TBS tuned
per-instance, 300 trajectories x 2000 periods per instance, `return_raw_trajectories` logged.

Raw data: `wideadjb_n100000_sequential_comparison.json` (full per-trajectory costs, tuning info,
and the complete eval spec embedded for reproducibility).

## N-scaling ladder on this range

| N | win rate vs CDI | mean gap | Wilcoxon p |
|---|---|---|---|
| 20,000 | 32.0% (16/50) | +7.69% (loses) | 0.001 (significant loss) |
| 50,000 | 78.0% (39/50) | -5.04% (wins) | 0.00001 (significant win) |
| 100,000 | **81.0% (81/100)** | **-5.46% (wins)** | **3.3e-12 (significant win)** |

**Diminishing returns are clear:** 20k->50k gained 46 points of win rate; 50k->100k gained only
3, despite 2x the training data (and using twice the test-instance sample size, n=100 vs n=50,
so the 100k figure is if anything more precisely estimated, not noisier). This looks like
convergence toward a ceiling around 80-82% for this range/architecture rather than a
sample-size-limited gap that more N would keep closing. The user's original target was a ~90%
win rate; this run does not reach it, and the trend suggests further N alone is unlikely to
close the remaining gap efficiently.

## Asymmetric win/loss magnitudes (important nuance beyond the raw win rate)

| | value |
|---|---|
| worst loss (max gap) | +3.67% |
| best win (min gap) | -52.47% |
| std of gap | 8.30% |

Even on the 19% of instances where GCA-DS does not beat CDI, it never loses by more than 3.7%.
Meanwhile several instances show 20-52% cost reductions. The strongly significant p-value
(3.3e-12) despite "only" 81% win rate is explained by this asymmetry: losses are small and
capped, wins are frequently large. Worth foregrounding in the paper alongside the win-rate
number - "wins big, loses small" is arguably the more informative summary than the win rate
alone.

## What characterizes the losing 19%?

| Parameter | correlation with gap (positive = more likely to lose) |
|---|---|
| b (backlog cost) | **+0.351** |
| \|c_e - c_r\| (cost differentiation between sources) | **-0.297** |
| l_r - l_e (lead time gap) | +0.176 (weak) |
| mu (demand rate) | -0.024 (~none) |

Losers (n=19) vs overall (n=100) means: b 2.605 vs 2.210 (higher), c_e 1.490 vs 2.148 (lower),
|c_e-c_r| 0.559 vs 1.016 (much lower).

**Interpretation:** GCA-DS's advantage is concentrated in instances with *low* backlog cost and
*large* cost differentiation between the expedited and regular sources - exactly where there is
real value in learning nuanced dual-sourcing timing, since a bigger cost gap between sources
means getting the sourcing decision right/wrong matters more, and CDI's fixed two-parameter
policy has more room to be suboptimal. Conversely, when b is high and c_e approx= c_r (little
benefit to differentiating sources), CDI is already close to as good as it can be, so there is
less headroom for GCA-DS to add value - and the instance's baseline cost tends to be smaller,
inflating the relative-gap noise. This is consistent with (and extends) the b-correlation found
earlier at N=20000/50000 on this same range (0.466 -> 0.448 -> 0.351): the correlation weakens
somewhat with more training data but does not disappear, i.e. more data raises the floor rather
than eliminating the underlying difficulty gradient.

Worst 5 losses (all small, <=3.7%): concentrated at high b (1.86-3.85) and/or low |c_e-c_r|
(0.0-1.32). Best 5 wins (all large, 24-52%): concentrated at low b (0.16-1.13) with meaningful
|c_e-c_r| (0.5-3.02).

## Implication for the paper

Recommend reporting this range's result as: "significantly and consistently better than CDI
(81% win rate, p<1e-11), with an asymmetric win/loss profile (worst loss +3.7%, best win -52%),
and a characterizable difficulty gradient (high backlog cost + low cost differentiation between
sources = harder for GCA-DS to add value, though it still never loses by more than a few
percent)" rather than the originally-hoped-for blanket "beats CDI 90% of the time" claim. This is
a more nuanced but arguably a more scientifically honest and interesting result.

## Raw data

`wideadjb_n100000_sequential_comparison.json` (100 instances x 5 policies, full per-trajectory
costs and CDI/DI/SI/TBS tuning info included).
