# Large-scale statistical CDI comparison: small range (RE-VALIDATION)

Date: 2026-08-13/14

## Why this test exists

The wide-adjusted-b range's earlier curated-set validation (10 stratified instances, monitored
live during training) showed the network beating CDI at every generation (overall mean -6.77%).
When properly re-tested against 50 genuinely random instances, that result did **not** hold up:
mean gap +7.69%, win rate only 32%, Wilcoxon p=0.001 significant **loss** to CDI (see
`large_scale_statistical_comparison_notes.md`). This raised the question: does the small range's
earlier apparent win (N=100000, overall mean -1.63% on its own curated 10-instance set) actually
hold up the same way, or was it also an artifact of an unrepresentative validation set?

## Methodology

Same protocol as the wide-range test: evaluated the already-trained small-range N=100000 final
policy (`sequential_mdp_config_k2_small_range_dcl_config_k2_range_n100000_20260812_195004`)
against 50 freshly-sampled random instances (different seed than any curated set used during
training), drawn uniformly from the training range: mu in [4,8], b in [5,15], c_e/c_r both in
[0.1,1.0] with c_r <= c_e, l pairs uniform over valid (l_e,l_r) with 1<=l_e<l_r<=3, h in [0.7,1.3]
(this range varies h too, unlike the wide range's fixed h=1.0 - `large_scale_cdi_comparison.py`
was extended with an `--h_range` option to support this). CDI tuned per-instance via the same
coordinate-ascent line search. Evaluation: `eval_k2_fixed`, 300 trajectories x 2000 periods.

## Result: CONFIRMED - the small range's win is real and statistically robust

| Metric | Value |
|---|---|
| n | 50 |
| mean gap vs CDI | **-4.97%** (negative = network wins) |
| std | 3.31% |
| win rate (network beats CDI) | **46/50 (92.0%)** |
| min (best win) | -10.76% |
| max (worst loss) | **+1.17%** |
| Wilcoxon signed-rank p-value (H0: median gap = 0) | **~0.000000** |

**This is a genuinely strong, publication-quality result.** Unlike the wide range, the small
range's advantage is real: it wins on 92% of random instances, the worst-case loss across all
50 draws is only 1.17% (compare to the wide range's 86% worst-case loss), and the improvement is
highly statistically significant in the correct direction. The earlier curated 10-instance
result (-1.63%) actually *understated* the true advantage somewhat (-4.97% on the proper random
sample) - so unlike the wide range, curated-set bias here happened to run conservative, not
inflated.

## What correlates with winning (or losing)?

| Parameter | correlation with gap |
|---|---|
| b (backlog cost) | **-0.71** (higher b -> bigger win) |
| h (holding cost) | +0.33 (higher h -> smaller win/more loss) |
| \|c_e - c_r\| | +0.21 |
| mu (demand rate) | -0.14 |

**Notably, b's correlation is strong here but in the OPPOSITE direction from the wide range**
(-0.71 here vs +0.47 there). This is not a contradiction - the two ranges don't overlap in b
(small range: [5,15], wide range: [0,4]) and are different training runs/architectures at
different N - but it's an important finding in its own right: **the relationship between b and
relative performance vs CDI is not a universal monotonic rule; it depends on where in parameter
space and at what N the network was trained.** This should be reported carefully in the paper -
avoid claiming "higher b is always harder/easier," and instead frame each range's finding as
specific to that trained model and range.

## Implication

The small range (mu[4,8], b[5,15], c[0.1,1], l[1,3], h[0.7,1.3]) with N=100000 is now the first
and only configuration in this project with a **rigorously validated, statistically significant,
practically meaningful** win over CDI across a genuinely random sample of the training range.
This is the strongest candidate for a "we beat CDI" claim in the paper. The wide range at
N=20000 does not support this claim; N=50000 on the wide range is still being evaluated the same
rigorous way to see if it closes the gap.

## Raw data

`large_scale_smallrange_n100000_comparison.csv` (50 rows).
