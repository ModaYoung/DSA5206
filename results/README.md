# Which results to use

The current report and repository README use the expanded analysis in `expanded/`.

- `expanded/all_models_summary.csv` combines the initial and expanded configurations.
- `expanded/figures/` contains the five current report figures.
- `expanded/half/` is the main 50-split comparison.
- `expanded/fivefold_pooled_summary.csv` is the fivefold sensitivity analysis.

The CSVs and two figures directly in this directory preserve the initial 900-fit comparison. They answer the narrower question of changing the regression head on the original DenseNet features. Its negative result is superseded in scope by the expanded experiment, which also evaluates newer encoders; it is retained as part of the audit trail.
