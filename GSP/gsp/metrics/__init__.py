"""Per-run metrics of PLAN §1.7 (S4: `state`; S5: the rest).

state          MetricContext: energy, AR_F, p_feas, eps_tilde, p_opt, p_top10, p_sector of a state
quality        AR_best_S (exact and Monte-Carlo), the best-of-S distribution, P(optimum seen)
convergence    effort to 0.95 AR_F(final), in circuits and two-qubit gate executions
resources      per unit / per circuit / to convergence, from counts.json
preprocessing  the sector-file join (GA cost, selection loss)
simdiff        F(chi), chi*, memory, half-cut entropy, peak RSS (numpy / quimb)
postrun        the post-run step: replay, 1000-shot sample (samples.npz), postrun.json
aggregate      `gsp aggregate` -> results/tables/metrics.parquet
checks/report  the S5 evidence and reports/metrics.md
"""
