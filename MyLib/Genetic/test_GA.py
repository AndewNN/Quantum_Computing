import numpy as np
import ga_solver
import time

num_assets = 9
budget = 380.17033271846947

asset_bit_lengths = [(3 if i % 3 == 0 else 2) for i in range(num_assets)]
asset_bit_lengths = np.ascontiguousarray(asset_bit_lengths, dtype=np.int32)

# price = [95.37000275, 169.58999634, 132.3500061, 109.72000122, 103.02999878, 178.19000244]
prices = [131.30000305, 165.86000061, 183.25999451, 168.47000122, 152.94000244, 95.37000275, 110.30999756, 176.30000305, 103.02999878]
prices = np.ascontiguousarray(prices, dtype=np.float64)

population_size = 1000
generations = 100
mutation_rate = 1.5 / sum(asset_bit_lengths)
crossover_rate = 0.85
elitism_count = 2
tournament_size = 5

st = time.perf_counter()
ga = ga_solver.GeneticAlgorithm(
    prices=prices,
    asset_bit_lengths=asset_bit_lengths,
    budget=budget,
    population_size=population_size,
    mutation_rate=mutation_rate,
    crossover_rate=crossover_rate,
    elitism_count=elitism_count,
    tournament_size=tournament_size
)
ga.run(generations, verbose=False)
top_inv = ga.get_top_n_individuals(24, False)
et = time.perf_counter()
# print(f"GA Execution Time: {(et - st) * 1000:.2f} ms")

best_solution = ga.get_best_individual()
# print("Actual distinct solutions found: ", len(top_inv))
total_bits = sum(asset_bit_lengths)

quantities = ga_solver.decode_chromosome_to_quantities(best_solution.chromosome, asset_bit_lengths, total_bits)

quantt = ga_solver.decode_chromosome_to_quantities(top_inv[23].chromosome, asset_bit_lengths, total_bits)

top_inv_gt = ga.get_top_n_brute_force_individuals(24, False)

quantities = np.array(quantt)
budd = np.dot(quantities, np.array(prices))
print(np.abs(budd - budget) / budget)