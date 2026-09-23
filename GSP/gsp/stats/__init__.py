"""Rules D1, D2, C1, C2 (PLAN §1.7; S3 built C1's operator level, S5 the rest).

d1         budget truncation, median-of-R / best-of-R, c*(B), q averaging, delta*, exact Wilcoxon + Holm, outcomes
d2         LOCO log-linear models, fold bootstrap, exponents, preconditions, verdict
c1         operator level (S3); leakage, eps_num calibration, state-level rule, growth-exponent fit (S5)
c2         KL to Haar on the reachable subspace, bootstrap, completion
ranktests  the exact signed-rank test (ties, zeros) and Holm
synthetic  data with a known answer for all four
"""
