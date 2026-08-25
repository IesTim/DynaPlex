# Online demand-rate estimation: cost of not knowing mu, and its convergence

Date: 2026-08-24/25

## Purpose

Quantify the performance cost of the online mu_hat estimator (Methodology Sec.5) versus
knowing the true demand rate mu from period 0, using the same trained policy
(sequential, N=100000, wide-adjusted-b) in two evaluation modes via the `oracle_mu_init`
flag - no retraining needed, since mu_hat/sigma_hat were already the network's only
demand-rate features (see mdp.cpp:258-262, 452-459).

## Main comparison (standard protocol: 300 traj x 2000 periods, 50 random instances, seed 2024)

| Comparison | mean gap |
|---|---|
| estimated mu vs. oracle mu | **+0.167%** |
| estimated mu vs. CDI | -5.87% |
| oracle mu vs. CDI | -6.02% |

At the standard evaluation horizon, the cost of having to estimate mu online rather than
knowing it exactly is negligible - 0.17%, essentially noise-level, against a policy that
beats CDI by ~6% either way. mu_hat converges fast enough within a 2000-period trajectory
that the estimation penalty barely registers by the time it's averaged over the whole
horizon.

## Horizon sweep (20 random instances, seed 2024, warmup=0 so the full trajectory from
period 0 is measured)

| periods_per_trajectory | mean gap, estimated vs. oracle |
|---|---|
| 200 | +2.898% |
| 500 | +1.393% |
| 1000 | +0.790% |
| 2000 | +0.435% |
| 5000 | +0.206% |

**Clean, monotonic convergence to zero as the horizon (and hence the number of demand
observations informing mu_hat) grows.** This is exactly the "estimation penalty vanishes
as observations accumulate" result the experiment was designed to show, and it does so
cleanly across a 25x range of horizon lengths without any non-monotonic noise. Good
candidate for a simple line-plot figure (horizon on x-axis, gap on y-axis, both log or
linear scale).

## Bottom line for the paper

Two results, both favorable: (1) at the horizon used throughout the rest of this thesis'
evaluation (2000 periods), the estimation penalty is negligible (~0.17%) relative to the
policy's ~6% advantage over CDI; (2) the penalty shrinks monotonically as the horizon
grows, from ~2.9% at 200 periods down to ~0.2% at 5000, confirming the mu_hat estimator
converges fast enough in practice that "not knowing mu in advance" is not a meaningful
practical concern for this policy.

## Raw data

`mu_estimation_oracle_vs_estimated.json` (main comparison, 50 instances x 3 policies),
`mu_estimation_horizon_{200,500,1000,2000,5000}.json` (20 instances x 2 policies each).
