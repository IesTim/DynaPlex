# Draft paper plan — 13 days to Sept 1

Living checklist. I'll update this file and re-send it as we go — check items off,
tell me your blocked days, and I'll rebalance the schedule around them.

## Scope decision (locked in, revisit only if something breaks)

- **No N=500,000 run.** Estimated 8-13 days on this machine — would eat the entire
  runway on one training job. Not required: you already said a smaller N with a
  statistically significant win over CDI is sufficient, and we have that.
- **Two headline results, already validated, become the paper's core empirical claims:**
  - Small range (mu[4,8], b[5,15], c[0.1,1], l[1,3], h[0.7,1.3]), N=100,000:
    92% win rate vs CDI, mean gap -4.97%, Wilcoxon p≈0.000000.
  - Wide-adjusted-b range (mu[2,12], b[0.1,4], c[0.1,4], l[1,5], h=1.0), N=50,000:
    78% win rate vs CDI, mean gap -5.04%, Wilcoxon p=0.00001.
- Everything below builds evidence, figures, and writing around these two results —
  no more open-ended N-scaling experiments.

## Day-by-day

Work-blocked days: Thu Aug 22, Fri Aug 23 (confirmed). Next week's 2 blocked days
not yet specified — placeholders below, tell me the real dates and I'll shift the
unattended background jobs onto those evenings instead.

| Day | Date | Focus | Notes |
|---|---|---|---|
| 1 | Aug 20 (today) | ✅ Branch cleanup + 4 logical commits done | Dropped dead build-artifact config duplicate + 2 stray dirs; pushed for review |
| 2 | Aug 21 | Kick off tightened statistical validation (n=100-150) as background jobs; start Methods section | Runs unattended overnight |
| 3 | Aug 22 (work — evening only) | Background jobs continue; evening: Methods section | |
| 4 | Aug 23 (work — evening only) | Background jobs finish; evening: prep benchmark-table regen | |
| 5 | Aug 24 | Regenerate benchmark table (DI/SI/TBS/CDI/GCA-DS) | Fix stale tuned_heuristic_params.json/eval_config wiring |
| 6 | Aug 25 | Convergence + robustness/OOD figures | Reuses existing per-generation CSVs, no new training |
| 7 | Aug 26 | Results section draft — plug in final numbers/figures | |
| 8 | Aug 27 *(placeholder — TBD blocked day?)* | Discussion + limitations section | |
| 9 | Aug 28 | Buffer / catch-up day | Absorb anything that overran |
| 10 | Aug 29 *(placeholder — TBD blocked day?)* | Introduction / related work | |
| 11 | Aug 30 | Full draft assembly, consistency pass on numbers/claims | |
| 12 | Aug 31 | Read-through + tightening | |
| 13 | Sep 1 | Final polish, submit | |

## Open questions for you

- Which 2 specific days next week are already blocked? (placeholders above at
  Aug 27 / Aug 29 — tell me the real dates and I'll rebalance)
- Any appetite for one more confirmatory training run (different seed) on either
  range for extra robustness, or are the two existing validated runs enough to
  stand on for the draft?
