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

## Raw data

`structural_bigwin_trace.json`, `structural_hardcorner_trace.json` (2000 periods each, full
per-period action pairs for both policies).
