"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for the S2 chromosome-layout tests.

Provenance
----------
source : CUDA/PO_new_ApproxRatio.py (repo Quantum_Computing, committed at 8634ca8,
         sha256 2979ad15585cdbe8a67eff72bee26214f7df8d5d64c40d242c14f396f68fe68b)
lines  : 516-526  GA chromosomes -> qubit strings (each asset block reversed)
source : MyLib/Genetic/genetic_solver.cpp (committed at 8634ca8,
         sha256 6e6744aa22f9872d39be5434645c18b2c13b5cb9c2af9337655415b7caf3d61a)
lines  : 254-270  calculate_fitness: quantity decode (MSB first) and total cost, transcribed to
                  Python line by line (C++ -> Python syntax only)
copied : 2026-09-24 (GSP session S2).
changes: the call-site lines are wrapped in `feasible_reversed_basis(...)` and dedented; the C++
         lines are transcribed into `cpp_total_cost(...)`; nothing else is changed.
"""


def feasible_reversed_basis(feasible_chromosomes_appr, N_ASSETS, TARGET_QUBIT_IN):
    feasible_reversed_basis_appr = []
    for i in range(len(feasible_chromosomes_appr)):
        chrom = feasible_chromosomes_appr[i]
        str_b = ""
        for aa in range(N_ASSETS):
            str_a = ""
            for c in range(TARGET_QUBIT_IN):
                str_a = str(int(chrom[aa * TARGET_QUBIT_IN + c])) + str_a
            str_b += str_a
        feasible_reversed_basis_appr.append(str_b)
    return feasible_reversed_basis_appr


def cpp_total_cost(chromosome, asset_bit_lengths, prices, budget):
    """(total_cost, fitness) of GeneticAlgorithm::calculate_fitness (budget mode)."""
    current_total_cost = 0.0
    curr_bit_idx = 0
    for i in range(len(prices)):
        quantity = 0
        for b in range(asset_bit_lengths[i]):
            quantity = (quantity << 1) | int(chromosome[curr_bit_idx + b])
        current_total_cost += quantity * prices[i]
        curr_bit_idx += asset_bit_lengths[i]
    diff = current_total_cost - budget
    return current_total_cost, diff * diff
