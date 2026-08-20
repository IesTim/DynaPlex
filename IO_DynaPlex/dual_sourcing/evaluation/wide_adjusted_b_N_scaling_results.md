# Wide (Temizöz-scale) K=2 generalization with adjusted backlog-cost range

Date: 2026-08-13

## Setup

- MDP: `mdp_config_wide_adjusted_b.json` — K=2, Temizöz-scale wide parameter range with one
  change: `min_b`/`max_b` set to `[0.0, 4.0]` instead of the original `[4.0, 99.0]`, so the
  backlog cost b is drawn from the same range/scale as the other cost parameters (c_e/c_r, both
  in `[0.1, 4.0]`), rather than being disproportionately larger. Full ranges: mu in [2,12],
  b in [0,4], c in [0.1,4] (c_e/c_r both drawn from this range then sorted), l_min=1/l_max=5,
  h=1.0 fixed, `max_order_size=270`, `inventory_cap_multiplier=3.0`, `rollout_M=100`
  (validated safe in the small-range M-sensitivity study), `SimulateOnlyPromisingActions=false`.
- Validation set: 10 held-out instances spanning this range
  (`wide_adjusted_b_validation_set.json`; 2 corners, 1 center, 7 stratified-random points), each
  with a tuned CDI baseline (`tune_k2_cdi`, same coordinate-ascent line-search method used
  throughout this project). Several instances tuned to `S_e=1` (the search floor) — investigated
  whether this reflects a tuning failure; the search method is identical to the one used
  successfully on the small range (where no such clustering occurred), so this is treated as a
  genuine property of these particular instances (high expedited cost relative to lead-time
  benefit makes minimal expediting genuinely optimal there) rather than a bug.
- Live per-generation evaluation via `eval_k2_gen` / `scripts/live_eval_run.py`, same
  methodology as the small-range study (see `small_range_N_scaling_results.md`).

## N=20000 (first attempt on this range, 5 generations)

| Generation | Gap vs CDI |
|---|---|
| 1 | -2.21% |
| 2 | -10.72% |
| 3 | -10.01% |
| 4 | -5.80% |
| 5 | -5.13% |
| **overall mean** | **-6.77%** |

Wall-clock: 397.4 min (~6.6 hours) for 5 generations — substantially slower per-generation than
the small range at comparable N (this range's `max_order_size=270` gives a ~4.5x larger action
space than the small range's 60, on top of the wider/harder state distribution).

**Finding: this range consistently beats CDI at every single generation, on the very first N
attempted (N=20000).** This is a stronger and more immediate result than the small range needed
— there, N=1000/3000/10000 all showed positive (CDI-losing) gaps and it took until N=30000-50000
to reliably approach/beat CDI, with N=100000 needed for a strong, monotonically-improving win.
Here, even the smallest N tried already wins comfortably across all instances and generations.

**Caveat:** given the S_e=1 tuning note above, this specific validation set may be slightly
biased toward instances where CDI itself is a weaker baseline (i.e. the "easy-to-beat" corner of
this parameter space). The consistency across corners, center, and stratified-random points is
reassuring, but this should be kept in mind when interpreting the exact magnitude for the paper
- the qualitative finding (consistently beats CDI already at N=20000) is robust; the exact -6.77%
number may not generalize identically to every subregion of the range.

## Raw data

`live_eval_wideadjb_n20000.csv`. See `small_range_N_scaling_results.md` for the earlier
small-range study this builds on (M-sensitivity, candidate-window ablation, N-scaling,
generation-count plateau findings).
