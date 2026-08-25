# Cost of zero-shot generalization: specialized vs. range-trained policy

Date: 2026-08-25

## Purpose

Directly quantify the payoff for this thesis' central trade-off: a single generally
capable, zero-shot-generalizing policy avoids retraining per instance (impractical - the
range-trained N=100000 run alone took 33.9h), but by construction it must spread its
capacity across an entire distribution of instances rather than specializing on one. This
experiment measures what that costs on one concrete instance.

## Methodology

Single fixed instance (mu=7, sigma=3.5, h=1, b=2, c_e=2.5, c_r=0.5, l_e=2, l_r=4 - the
same baseline instance used as the OOD sweep's center point, for coherence), trained from
scratch with the parameter range collapsed to this one point
(`mdp_config_wideadjb_zeroshot_baseline.json`), same N=50000 and network architecture as
one rung of the main N-scaling ladder (`dcl_config_k2_wideadjb_n50000.json`), sequential
action representation. Compared against the range-trained N=100000 policy and tuned CDI,
all evaluated on this same instance (300 traj x 2000 periods).
Run: `sequential_mdp_config_wideadjb_zeroshot_baseline_dcl_config_k2_wideadjb_n50000_20260824_203239`,
5 generations, 16h34m wall clock (much faster than the range-trained runs, consistent with
single-instance learning being a substantially easier problem than range generalization).

## Result

| Policy | cost | gap vs CDI |
|---|---|---|
| zero-shot specialized (N=50000, this instance only) | 16.7467 | **-3.56%** |
| range-trained (N=100000, full wide-adjusted-b range) | 17.0496 | -1.82% |
| CDI (tuned for this instance) | 17.3649 | - |

**Cost of generalization: the specialized policy beats the range-trained policy by 1.78%
on this instance.** Both beat CDI comfortably. This is a modest, clearly-bounded price for
the practical benefit of not needing to retrain per instance - and notably, the
specialized policy reaches this with *less* training data (N=50000 vs. N=100000),
consistent with single-instance learning being an easier problem that saturates faster
than learning across a whole parameter range.

## Caveat

n=1 instance, per the explicit instruction to start with one and extend only if time
allows. This single data point cannot establish whether 1.78% is representative of the
"typical" cost of generalization across the range, or whether it is unusually small/large
for this particular instance (e.g. it sits centrally in the range, not at a hard corner
like high-b/low-cost-differentiation instances identified elsewhere as harder for
GCA-DS). A second instance at a harder corner would meaningfully strengthen this result if
time allows; recommend not overstating generality from n=1 in the write-up.

## Raw data

`zeroshot_cost_of_generalization.json` (1 instance x 3 policies, full per-trajectory
costs).
