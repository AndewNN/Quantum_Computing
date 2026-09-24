"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for tests/test_dbqite.py (PLAN §5 S8: A4 / A6 must match
this script step by step on its 3-qubit instance: energies and the chosen s).

Provenance
----------
source : ~/Desktop/Quantum_Master_Proposal/Lecture_Notes/code/example_dbqite.py (outside this repo, read-only),
         sha256 c6c0279d5793d24fbf39a6454185a73933861a5a812f76c92acf292fe422bd51 (31 lines)
lines  : 1-31, the whole file
copied : 2026-09-24 (GSP session S8), byte-for-byte after this header (the header is the only addition).
         tests/test_dbqite.py checks that the body below equals the source file when it is present, and runs it by
         `exec` with its prints captured.
"""
import numpy as np

d = 8                                           # 3 qubits
levels = np.array([-2.0, -1.2, -0.7, -0.1, 0.3, 0.8, 1.4, 2.1])   # energies of the 8 bitstrings
H = np.diag(levels).astype(complex)             # diagonal cost Hamiltonian
def U(Hm, t):                                   # exp(-i Hm t), Hermitian Hm
    w, V = np.linalg.eigh(Hm)
    return (V * np.exp(-1j * w * t)) @ V.conj().T
energy = lambda psi: float(np.real(psi.conj() @ H @ psi))

def compiled_step(psi, s):                      # what the circuit applies: the group commutator
    rho, r = np.outer(psi, psi.conj()), np.sqrt(s)
    # e^{irH} e^{ir rho} e^{-irH} e^{-ir rho} |psi>,  with  U(A, t) = e^{-iAt}
    return U(H, -r) @ U(rho, -r) @ U(H, r) @ U(rho, r) @ psi

psi = np.ones(d, dtype=complex) / np.sqrt(d)    # |+++>
grid = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]    # candidate step sizes
for k in range(7):
    rho = np.outer(psi, psi.conj())
    W = rho @ H - H @ rho                        # anti-Hermitian generator [rho, H]
    E = energy(psi); V = float(np.real(psi.conj() @ H @ H @ psi)) - E**2
    # the one greedy scalar choice, made on the compiled step because that is what a device runs
    s = grid[int(np.argmin([energy(compiled_step(psi, x)) for x in grid]))]
    exact = U(1j * W, s) @ psi                   # exp(sW) = exp(-i s (iW)), the ideal step
    new = compiled_step(psi, s)
    print(f"k={k}  E={E:+.4f}  V={V:.4f}  s={s:<4}  -2sV={-2*s*V:+.4f}"
          f"  ideal step {energy(exact)-E:+.4f}  compiled step {energy(new)-E:+.4f}"
          f"  p_ground={abs(new[0])**2:.3f}")
    psi = new
print("ground energy", round(float(levels[0]), 4))
print("copies of U_0 in the circuit after 7 steps:", 3**7)
