# Structural analysis: where does sequential GCA-DS choose differently from CDI?

Date: 2026-08-26

## Purpose

Test 7 from the original experiment list (lowest priority, attempted now that everything else is
done): does GCA-DS make systematically different sourcing decisions than CDI, and if so, is there
an economic interpretation - i.e. does the difference look like GCA-DS has learned something
genuinely smarter than CDI's fixed-threshold structure, rather than just noise?

## Methodology

Directly motivated by the correlation finding in `wideadjb_n100000_sequential_statistical_comparison_notes.md`:
GCA-DS's advantage over CDI correlates with low backlog cost `b` (corr +0.35) and large cost
differentiation `|c_e-c_r|` (corr -0.30) - i.e. it wins big in low-b/high-differentiation
instances and only narrowly in high-b/low-differentiation ones. Picked one instance from each end
of that earlier "best 5 wins" / "worst 5 losses" list, and traced 2000 periods of both policies'
actions along the same CDI-driven trajectory (added `dual_sourcing_eval`'s new `structural` mode
for this - drives one trajectory under CDI's own actions so the system stays in realistic states,
and at every period's two sub-decisions records what CDI actually chose *and* what GCA-DS would
have chosen from that exact same state, following the pattern already validated in
`dual_sourcing_validate.cpp`'s `DebugTraceK2Actions`).

- **"Big win" instance**: mu=9.93, b=0.16 (very low backlog cost), c_e=3.52, c_r=0.50 (7x price
  ratio between sources), l=(3,4). Overall gap on this instance in the earlier 100-instance test:
  -52.47% (GCA-DS's best win in that sample).
- **"Hard corner" instance**: mu=11.79, b=3.85 (high backlog cost), c_e=2.32, c_r=1.49 (only 1.56x
  price ratio), l=(1,2). Overall gap: +3.67% (GCA-DS's worst loss in that sample, though still
  small in absolute terms).

## Results

| | Big win instance | Hard corner instance |
|---|---|---|
| mean q_expedited, GCA-DS | **0.00** | 0.81 |
| mean q_expedited, CDI | 0.61 | 1.05 |
| frequency GCA-DS orders expedited at all | **0.0%** | 20.8% |
| frequency CDI orders expedited at all | 14.5% | 27.4% |
| mean q_regular, GCA-DS | 3.28 | 11.25 |
| mean q_regular, CDI | 9.53 | 10.96 |
| corr(GCA-DS q_regular, CDI q_regular) | 0.795 | **0.963** |
| corr(GCA-DS q_expedited, CDI q_expedited) | undefined (GCA-DS constant 0) | **0.982** |

## Interpretation

**In the big-win instance, GCA-DS has learned to essentially abandon the expedited source
entirely** - never using it once across 2000 periods, versus CDI still reaching for it 14.5% of
the time. This is economically sensible, not arbitrary: with backlog cost this low (b=0.16) and
the expedited source this much pricier (7x), the rational response to an emerging shortage is
"just wait it out" rather than pay a steep premium for speed - being briefly backlogged costs
almost nothing here, so there is little to gain from expediting. CDI's fixed order-up-to
thresholds (S_e=1, tuned but still a static rule) hedge against shortage regardless of how cheap
that shortage actually is to tolerate, and are the wrong shape for this specific trade-off. GCA-DS
also orders from the cheap regular source noticeably less often and in smaller quantities (3.28
vs. 9.53 mean, 49.1% vs. 98.9% of periods) - consistent with tolerating more backlog rather than
holding costly safety stock when holding cost matters more than backlog cost.

**In the hard-corner instance, GCA-DS's actions track CDI's almost exactly** (correlation 0.96-0.98
for both sources, nearly identical mean order quantities). This is a genuinely useful finding on
its own, not just "no difference to report": it explains *why* GCA-DS's advantage narrows in this
region of the parameter space (already established statistically in the earlier N=100000 result)
- not because GCA-DS is struggling, but because when backlog is expensive and the two sources cost
nearly the same, CDI's simple "keep inventory topped up from both sources" heuristic is already
close to whatever the right policy looks like. There is little room for a more nuanced policy to
improve on a heuristic that is already nearly correct in this regime.

**Taken together, this gives the correlation finding from the N-100000 statistical result a
genuine causal/mechanistic story**, not just an observed pattern: GCA-DS's advantage is largest
exactly where CDI's fixed-threshold structure is the worst fit for the true cost trade-off (cheap
backlog, expensive differentiation - GCA-DS learns to selectively ignore an option CDI still
hedges with), and smallest where CDI's structure already happens to be close to right (expensive
backlog, similar costs - both policies converge to similar, aggressive dual-sourcing behavior).

## Extension: systematic 32-instance grid (2026-08-27)

The two-instance comparison above is a proof of concept - it shows the pattern exists at two
extremes, but not whether it's a real, continuous relationship or an artifact of two convenient
picks. Extended `dual_sourcing_eval structural` to also compute the aggregate performance gap
(`EvaluatePolicyTuning` on both policies, same instance) alongside the action trace, then ran a
full grid: `b` in {0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0} x `c_e` in {0.75, 1.5, 2.5, 3.5} (`c_r`
fixed at 0.5), `mu`=8, `l`=(2,4) held fixed - 32 instances, 1500-period trace + a lighter
100 traj x 1000 period evaluation each (each instance took ~5-7 seconds; the whole grid finished
in about 3 minutes). Spec generator + raw grid files: `eval_spec_structural_grid_*.json` /
`structural_grid_*.json`; summary table extracted to `/tmp/structural_grid_summary.json` (not
committed - regenerable from the 32 raw files).

### The key result: structural divergence predicts performance advantage, not just accompanies it

**corr(expedited-usage gap [GCA-DS minus CDI], performance gap vs. CDI) = 0.672** across all 32
instances. The more GCA-DS's behavior diverges from CDI's - specifically, the more *less* often it
uses the expedited source than CDI does - the bigger GCA-DS's cost advantage over CDI. This is the
finding the two-instance version could only illustrate; the grid actually establishes it as a
real, quantified relationship. See `structural_usage_vs_performance.pdf`.

### The full picture is richer than the two-instance version suggested

- **corr(b, performance gap) = 0.664**, **corr(b, GCA-DS expedited-usage frequency) = 0.470**,
  **corr(cost differentiation, GCA-DS expedited-usage frequency) = -0.736**. Cost differentiation
  drives *how often GCA-DS uses the expensive source* (strong, as expected); `b` drives both usage
  frequency and the performance gap, but more weakly than differentiation drives usage.
- **At very low b (0.1), GCA-DS abandons the expedited source completely regardless of price gap**
  (0.0-0.2% usage across all four `c_e` values tested), while CDI still uses it 22-61% of the time.
  This generalizes the original "big win" instance from a single anecdote to an entire row of the
  grid: near-zero backlog cost reliably triggers complete abandonment of the fast source, not just
  in the one instance originally picked.
- **A genuine surprise the two-instance version could not have shown: at moderate-to-high b with a
  *small* price gap (c_e=0.75), GCA-DS actually uses the expedited source *more often* than CDI
  does** (usage gap turns positive: +2.3, +7.0, +8.0, +8.8, +7.7 percentage points for b=1.5
  through 4.0). GCA-DS is not simply "always more conservative than CDI" - when backlog is
  expensive and the two sources cost almost the same, hedging aggressively with *both* sources
  becomes the better strategy, and GCA-DS leans into that harder than CDI's fixed thresholds do.
  This nuance was invisible from the original 2-instance comparison and only shows up once the
  full (b, cost-differentiation) plane is mapped.
- **The relationship between cost differentiation and the performance gap is not simply monotonic
  once b is fixed** (corr(cost differentiation, performance gap) = -0.140 overall, much weaker
  than the b or usage-gap correlations) - e.g. at b=1.0 the gap goes -8.73% -> -4.94% -> -8.12% ->
  -15.70% as c_e increases from 0.75 to 3.5, not a straight line. The two variables interact rather
  than contributing additively; b is the more reliable single predictor of performance gap, cost
  differentiation is the more reliable single predictor of *how* GCA-DS's behavior diverges from
  CDI's.

### Bottom line for the paper

The grid upgrades the structural analysis from "here is an interesting anecdote at two points"
to "here is a real, quantified, 32-instance relationship between how differently the learned
policy behaves and how much better it performs" - the single strongest piece of evidence in the
whole thesis that GCA-DS's advantage over CDI is mechanistically explainable rather than
incidental. The b=0.1 row (complete abandonment of the expedited source, regardless of price gap)
and the small-price-gap/high-b corner (GCA-DS hedging *more* than CDI) are both worth featuring
explicitly - together they show GCA-DS adapting its qualitative strategy in both directions
depending on the instance, not applying one fixed correction to CDI's behavior.

## Raw data

`structural_bigwin_trace.json`, `structural_hardcorner_trace.json` (original 2-instance
comparison, 2000 periods each). `structural_grid_*.json` (32 files, the systematic b x cost_e
grid, 1500-period trace + performance gap each). Figures: `structural_expedited_usage.pdf` (the
original 2-instance bar chart), `structural_grid_heatmaps.pdf` (performance gap and usage gap
across the full grid), `structural_usage_vs_performance.pdf` (the key scatter establishing the
0.672 correlation).
