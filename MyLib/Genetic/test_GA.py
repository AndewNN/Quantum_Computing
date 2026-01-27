import os
import numpy as np
import ga_solver
import time
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-A', '--assets', 
    type=int, default=5, help='Number of assets to optimize'
)
args = parser.parse_args()
num_assets = args.assets

# num_assets = 5

if num_assets == 3:
    budget = 476.7820375549709
elif num_assets == 4:
    budget = 319.9832552190401
elif num_assets == 5:
    budget = 385.0336241146705
elif num_assets == 6:
    budget = 381.0143111171548
elif num_assets == 7:
    budget = 376.05610668210534
elif num_assets == 8:
    budget = 369.6004106238545
elif num_assets == 9:
    budget = 380.17033271846947
elif num_assets == 10:
    budget = 378.958463388318

asset_bit_lengths = [(3 if i % 3 == 0 else 2) for i in range(num_assets)]
asset_bit_lengths = np.ascontiguousarray(asset_bit_lengths, dtype=np.int32)

if num_assets == 3:
    prices = [176.30000305, 131.30000305, 136.53999329]
elif num_assets == 4:
    prices = [131.28999329, 129.33999634,  96.33999634, 103.02999878]
elif num_assets == 5:
    prices = [169.58999634, 131.30000305, 183.25999451,  96.33999634, 178.19000244]
elif num_assets == 6:
    prices = [95.37000275, 169.58999634, 132.3500061, 109.72000122, 103.02999878, 178.19000244]
elif num_assets == 7:
    prices = [109.72000122,  95.37000275, 169.58999634, 129.33999634, 132.3500061, 176.30000305, 131.28999329]
if num_assets == 8:
    prices = [103.02999878, 178.19000244, 132.3500061,  165.86000061, 152.94000244, 168.47000122,  96.33999634, 176.30000305]
elif num_assets == 9:
    prices = [131.30000305, 165.86000061, 183.25999451, 168.47000122, 152.94000244, 95.37000275, 110.30999756, 176.30000305, 103.02999878]
elif num_assets == 10:
    prices = [ 95.37000275, 129.33999634, 184.53999329, 109.72000122, 103.02999878, 132.3500061,  176.30000305, 183.25999451, 131.30000305, 131.28999329]
prices = np.ascontiguousarray(prices, dtype=np.float64)

# ------------ ASSETS = 5 ------------ #
if num_assets == 5:
    population_size = 2000
    generations = 35
    mutation_rate = 1.5 / sum(asset_bit_lengths)
    crossover_rate = 0.85
    elitism_count = 2
    tournament_size = 5

# ------------ ASSETS = 9 ------------ #
elif num_assets == 9:
    population_size = 2000
    generations = 35
    mutation_rate = 1.5 / sum(asset_bit_lengths)
    crossover_rate = 0.85
    elitism_count = 2
    tournament_size = 5

else:
    population_size = 2000
    generations = 35
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
GA_time = et - st
print(f"GA Execution Time: {GA_time * 1000:.2f} ms")

best_solution = ga.get_best_individual()
# print("Actual distinct solutions found: ", len(top_inv))
total_bits = sum(asset_bit_lengths)

quantities = ga_solver.decode_chromosome_to_quantities(best_solution.chromosome, asset_bit_lengths, total_bits)

quantt = ga_solver.decode_chromosome_to_quantities(top_inv[23].chromosome, asset_bit_lengths, total_bits)

st = time.perf_counter()
top_inv_gt = ga.get_top_n_brute_force_individuals(24, False)
et = time.perf_counter()
BF_time = et - st
print(f"Brute Force Execution Time: {BF_time * 1000:.2f} ms")

# quantities = np.array(quantt)
# budd = np.dot(quantities, np.array(prices))
# print(np.abs(budd - budget) / budget)

# print(budd)
# print(top_inv[23].total_cost)

for i in range(24):
    budd = top_inv[i].total_cost
    budd_gt = top_inv_gt[i].total_cost
    print(np.abs(budd - budget) / budget, np.abs(budd_gt - budget) / budget)

# print(top_inv[0].chromosome)
dirr = "./"
file_name = f"speed.csv"
file_path = f"{dirr}/{file_name}"
col = ["Assets", "GA_time_ms", "BF_time_ms", "mean12_eps_GA", "mean24_eps_GA", "mean12_eps_BF", "mean24_eps_BF"]

all_diff_ga, all_diff_bf = 0, 0
for i in range(12):
    budd = top_inv[i].total_cost
    budd_gt = top_inv_gt[i].total_cost
    all_diff_ga += np.abs(budd - budget) / budget
    all_diff_bf += np.abs(budd_gt - budget) / budget
mean12_eps_GA = all_diff_ga / 12
mean12_eps_BF = all_diff_bf / 12

for i in range(12, 24):
    budd = top_inv[i].total_cost
    budd_gt = top_inv_gt[i].total_cost
    all_diff_ga += np.abs(budd - budget) / budget
    all_diff_bf += np.abs(budd_gt - budget) / budget
mean24_eps_GA = all_diff_ga / 24
mean24_eps_BF = all_diff_bf / 24


df = (pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame(columns=col))
new_row = {
    "Assets": num_assets,
    "GA_time_ms": GA_time * 1000,
    "BF_time_ms": BF_time * 1000,
    "mean12_eps_GA": mean12_eps_GA,
    "mean24_eps_GA": mean24_eps_GA,
    "mean12_eps_BF": mean12_eps_BF,
    "mean24_eps_BF": mean24_eps_BF
}
if os.path.exists(file_path):
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
else:
    df = pd.DataFrame([new_row])
df.to_csv(file_path, index=False)
print(f"Done Asset: {num_assets}, Num: {len(df)}")