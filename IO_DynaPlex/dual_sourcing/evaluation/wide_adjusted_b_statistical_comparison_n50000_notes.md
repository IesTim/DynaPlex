# Wide-adjusted-b range: N=20000 vs N=50000 statistical comparison

Date: 2026-08-14

## Context

Note: "wide-adjusted-b range" refers to this project's own extension of Temizöz's DynaPlex
codebase to the dual-sourcing backlog problem (a different model from Temizöz's lost-sales
paper). The parameter ranges here (mu, b, c, lead times) are this project's own settings, not
values taken from Temizöz's paper. The only thing genuinely inherited from Temizöz's methodology
is the eventual target sample budget N=5,000,000 - see [[project-dual-sourcing-paper]] memory.

Range: mu in [2,12], b in [0.1,4] (b's range deliberately narrowed from an earlier [4,99] to
match the scale of the other cost parameters), c_e/c_r both in [0.1,4], l_min=1/l_max=5, h=1.0
fixed, max_order_size=270, inventory_cap_multiplier=3.0, rollout_M=100.

## The decisive question

N=20000's curated-set validation (10 stratified instances) showed the network beating CDI at
every generation. The proper 50-random-instance statistical test contradicted this: mean gap
+7.69%, win rate 32%, Wilcoxon p=0.001 (significant **loss**). Per the small-range precedent
(where N=100000 fixed an analogous gap), we tested whether more data (N=50000) fixes this range
too - using the exact same 50-instance random seed (2024) for a clean apples-to-apples
comparison.

## Result: YES - N=50000 reverses the failure into a genuine, statistically significant win

| Metric | N=20000 | N=50000 |
|---|---|---|
| mean gap vs CDI | **+7.69%** (loses) | **-5.04%** (wins) |
| std | 17.17% | 7.83% |
| win rate | 32.0% (16/50) | **78.0% (39/50)** |
| min (best win) | -29.43% | -29.75% |
| max (worst loss) | +86.32% | +14.45% |
| Wilcoxon p-value | 0.001 (significant loss) | **0.00001 (significant win)** |
| Training wall-clock | 6.6 hours | ~20 hours |

This is a complete reversal, not just an improvement at the margin: mean gap flips sign, win
rate more than doubles, variance roughly halves, and the worst-case loss drops from a
catastrophic 86% to a modest 14%. The improvement is highly statistically significant in both
directions (N=20000's loss and N=50000's win are each individually significant at p<0.01).

**Per the user's stated goal** (find the smallest N that gives a robust win, not necessarily
reach N=5,000,000): N=50000 already clears this bar convincingly. Further N-scaling on this
range is not required to support a "beats CDI" claim, though additional generations/seeds could
still tighten the confidence interval for the paper if desired.

## What correlates with losing (still)?

| Parameter | correlation with gap, N=20000 | correlation with gap, N=50000 |
|---|---|---|
| b (backlog cost) | +0.47 | **+0.45** (persists) |
| \|c_e - c_r\| | +0.18 | -0.21 (weak, flips sign) |
| mu (demand rate) | -0.16 | 0.02 (vanishes) |

**Notably, the b-correlation does not go away with more data** - it's essentially unchanged in
magnitude (0.47 -> 0.45). What changes is the overall level: at N=50000 even the harder
(high-b) instances mostly still win or land near break-even, whereas at N=20000 the same
instances were often losing outright. This suggests b genuinely does make instances harder to
learn precisely (consistent with the earlier "extremity"/fractile-precision hypothesis from the
small-range corner analysis), and more data compensates for that difficulty rather than
eliminating it. Worth reporting in the paper as: "gap-to-CDI is not uniform across the range;
harder (high-b) instances need more data to close, but the relationship holds at both N levels
we tested."

## Raw data

`large_scale_wideadjb_n50000_comparison.csv` (50 rows, same random instances/seed as the
N=20000 comparison for direct comparability).
