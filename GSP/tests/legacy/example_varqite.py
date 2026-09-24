"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for tests/test_varqite.py (PLAN §5 S7: A3 must match
this script on its 3-qubit instance).

Provenance
----------
source : ~/Desktop/Quantum_Master_Proposal/Lecture_Notes/code/example_varqite.py (outside this repo, read-only),
         sha256 70ca33a10a4acabc42ef17d1ff740a9675d8a9ad85f606da99a8469afbf04825 (53 lines)
lines  : 1-53, the whole file
copied : 2026-09-24 (GSP session S7), byte-for-byte after this header (the header is the only addition).
         tests/test_varqite.py checks that the body below equals the source file when it is present, and runs
         the part before the first print (the instance and `varqite`) by `exec`.
"""
import numpy as np

n = 3
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
I = np.eye(2)
def op(single, i):
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, single if k == i else I)
    return out
def U(H, t):                                   # exp(-i H t) for Hermitian H
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w * t)) @ V.conj().T

rng = np.random.default_rng(11)
J = rng.normal(size=(n, n)); h = rng.normal(size=n)
H_C = sum(J[i, j] * op(Z, i) @ op(Z, j) for i in range(n) for j in range(i + 1, n)) \
    + sum(h[i] * op(Z, i) for i in range(n))
H_X = sum(op(X, i) for i in range(n))
plus = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

def varqite(L, dtau=0.01, steps=400, eps=1e-6, report=False):
    gens = [H_C, H_X] * L                      # theta = (gamma_1, beta_1, ..., gamma_L, beta_L)
    p = len(gens)
    theta = 0.05 * np.random.default_rng(1).normal(size=p)   # not exactly 0: M is singular there
    for step in range(steps + 1):
        layers = [plus]
        for G, t in zip(gens, theta):
            layers.append(U(G, t) @ layers[-1])
        phi, d = layers[-1], []
        for k in range(p):        # |d_k phi> = U_p ... U_{k+1} (-i G_k) U_k ... U_1 |psi_0>
            v = -1j * gens[k] @ layers[k + 1]
            for j in range(k + 1, p):
                v = U(gens[j], theta[j]) @ v
            d.append(v)
        A = np.array([[np.real(d[i].conj() @ d[j]) for j in range(p)] for i in range(p)])
        a = np.array([np.imag(phi.conj() @ d[j]) for j in range(p)])    # <phi|d_j phi> = i a_j
        M = A - np.outer(a, a)                                          # global-phase-adjusted metric
        C = np.array([np.real(d[i].conj() @ H_C @ phi) for i in range(p)])
        E = np.real(phi.conj() @ H_C @ phi)
        if report and step % 100 == 0:
            V = np.real(phi.conj() @ H_C @ H_C @ phi) - E**2
            R = V - C @ np.linalg.solve(M + eps * np.eye(p), C)         # McLachlan residual
            print(f"  tau={step*dtau:3.1f}  E={E:+.4f}  V={V:6.3f}  residual={R:6.3f}"
                  f"  cond(M)={np.linalg.cond(M):.1e}")
        theta = theta - dtau * np.linalg.solve(M + eps * np.eye(p), C)   # Euler step of M thetadot=-C
    return E

print("L=2 in detail:"); varqite(2, report=True)
for L in [1, 2, 4, 6]:
    print(f"L={L}  p={2*L:2d}  final energy {varqite(L):+.4f}")
print("exact ground energy", round(float(np.min(np.real(np.diag(H_C)))), 4))
