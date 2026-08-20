# Draft paper plan — 14 days to Sept 1

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

Fill in your 4 already-blocked days below and I'll move compute-heavy (unattended)
background jobs onto those evenings.

| Day | Date | Focus | Notes |
|---|---|---|---|
| 1 | Aug 18 (today) | Clean up branch, commit, push | Drop dead config duplicate + stray dirs; split into logical commits; review via GitHub compare view |
| 2 | Aug 19 | Kick off tightened statistical validation (n=100-150) as background jobs | Runs unattended — good day to also start Methods section writing |
| 3 | Aug 20 | Methods section draft; check on background runs | |
| 4 | Aug 21 | *(blocked? evening only)* Regenerate benchmark table (DI/SI/TBS/CDI/GCA-DS) | |
| 5 | Aug 22 | *(blocked? evening only)* Convergence + robustness/OOD figures | Reuses existing per-generation CSVs, no new training |
| 6 | Aug 23 | Results section draft — plug in final numbers/figures | |
| 7 | Aug 24 | *(blocked? evening only)* Results section continued | |
| 8 | Aug 25 | Buffer / catch-up day | Absorb anything that overran |
| 9 | Aug 26 | Discussion + limitations section | |
| 10 | Aug 27 | *(blocked? evening only)* Introduction / related work | |
| 11 | Aug 28 | Full draft assembly, consistency pass on numbers/claims | |
| 12 | Aug 29 | Read-through + tightening | |
| 13 | Aug 30 | Buffer / final figure polish | |
| 14 | Sep 1 | Final polish, submit | |

## Open questions for you

- Which 4 specific days are already blocked? (I'll mark them evening-only above)
- Any appetite for one more confirmatory training run (different seed) on either
  range for extra robustness, or are the two existing validated runs enough to
  stand on for the draft?
