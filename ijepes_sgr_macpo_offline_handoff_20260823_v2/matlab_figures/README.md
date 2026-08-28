# MATLAB figure workspace

This directory is intentionally separate from the manuscript figures.  It
contains reproducible data exports, MATLAB sources, and candidate outputs.

- `scripts/export_behavior_data.py` exports the archived behavior report to CSV.
- `scripts/run_all_figures.m` generates the current candidate figures.
- The manuscript delay counterfactual plots electrolyzer power, stored hydrogen,
  and pending hydrogen so that the controlled timing change is visible in
  physical trajectories rather than only in daily aggregate bars.
- `data/recurrent_ablation_seed30.csv` and `scripts/plot_recurrent_ablation_seed30.m`
  provide the same-checkpoint recurrent-information ablation used in the case
  study. The values are the paper-authoritative approximate costs reported in
  the manuscript text.
- `scripts/evaluate_seed32_delivery_counterfactual.py` performs the no-training,
  same-checkpoint delayed/instant delivery comparison on the canonical server.
- `scripts/instant_delivery_counterfactual.patch` is the small evaluation-only
  environment patch required by that script. From the canonical HyperMARL root,
  apply it with `git apply <path-to-patch>`, run the evaluator, then restore the
  environment source. The archived source itself remains byte-identical to the
  downloaded server copy.
- `data/manifest.json` records the exact archived source.
- `output/` is reviewed before any manuscript figure is replaced.

The current manuscript is authoritative if a server-side log differs.  The
convergence curves are therefore preserved by cropping the current 600-dpi
manuscript image; they are not regenerated from mismatched training logs.
