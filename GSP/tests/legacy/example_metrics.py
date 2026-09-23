"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for tests/test_metrics.py (PLAN §5 S5: the metrics
test must reproduce this script to 1e-12).

Provenance
----------
source : ~/Desktop/Quantum_Master_Proposal/Lecture_Notes/code/example_metrics.py (outside this repo, read-only),
         sha256 d2eded7ad35103852552e1a9826306bfa3b9f2f719388f9f8a132817860803c2 (30 lines)
lines  : 1-30, the whole file
copied : 2026-09-24 (GSP session S5), byte-for-byte after this header (the header is the only addition).
         `tests/test_metrics.py` checks that the body below equals the source file when it is present.
"""
import numpy as np
from itertools import product
rng = np.random.default_rng(5)

# a toy instance: 2 assets, 2 bits each (n = 4), holdings v_i = x_i0 + 2 x_i1
P = np.array([0.35, 0.56])                                 # normalized prices P'
enc = np.array([[1, 2, 0, 0], [0, 0, 1, 2]])               # v = enc @ x
bits = np.array(list(product([0, 1], repeat=4)))
Delta = np.abs(bits @ enc.T @ P - 1)                       # budget violation of every bitstring
f = rng.normal(size=16)                                    # stand-in for the objective <x|H_obj|x>
eps = 0.1
feasible = Delta <= eps

psi = rng.normal(size=16) + 1j * rng.normal(size=16); psi /= np.linalg.norm(psi)
prob = np.abs(psi)**2

p_feas = prob[feasible].sum()
eps_tilde = np.sqrt(np.sum(prob * Delta**2))               # sqrt <psi|H_penalty|psi>
E_min, E_max = f[feasible].min(), f[feasible].max()
quality = (E_max - f) / (E_max - E_min)                    # 1 = best feasible, 0 = worst feasible
AR_F = np.sum(prob[feasible] * quality[feasible])
p_opt = prob[feasible & np.isclose(f, E_min)].sum()

S = 20                                                     # best feasible bitstring among S shots
shots = rng.choice(16, size=(20000, S), p=prob)
best = [quality[row[feasible[row]]].max() for row in shots if feasible[row].any()]
print(f"feasible bitstrings: {feasible.sum()} of 16")
print(f"p_feas = {p_feas:.3f}  eps_tilde = {eps_tilde:.3f}  AR_F = {AR_F:.3f}  p_opt = {p_opt:.3f}")
print(f"AR_best_S (S={S}) = {np.mean(best):.3f}")
print(f"P(optimum seen in S shots) = {1-(1-p_opt)**S:.3f}")
