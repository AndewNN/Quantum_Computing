import cudaq
from typing import List
import numpy as np
    
@cudaq.kernel
def kernel_oracle(v: cudaq.qvector, r: cudaq.qubit, l: List[int]):
    qb = len(v)
    for k in l:
        for i in range(qb):
            if int(k / (2**i)) % 2 == 0:
                x(v[qb - i - 1])
        x.ctrl(v, r)
        for i in range(qb):
            if int(k / (2**i)) % 2 == 0:
                x(v[qb - i - 1])

@cudaq.kernel
def kernel_diffuser(v: cudaq.qvector):
    qb = len(v)
    h(v)
    x(v)
    z.ctrl([v[0:qb-1]], v[qb-1])
    x(v)
    h(v)

@cudaq.kernel
def kernel_grover(qb:int, l: List[int], iterations: int):
    v = cudaq.qvector(qb)
    r = cudaq.qubit()
    h(v)
    x(r)
    h(r)
    for _ in range(iterations):
        kernel_oracle(v, r, l)
        kernel_diffuser(v) 
    mz(v)

qb = 7
nums = [68, 103, 119, 45]

zeta = np.arcsin(np.sqrt(len(nums) / (2 ** qb)))
print(f'zeta: {zeta / np.pi * 180:.2f}')

iters = round(np.pi / (4 * zeta) - 0.5)
print(f'\nOptimal iterations: {iters}')
optimal_zeta = zeta * (2 * iters + 1)
print(f'Optimal zeta: {optimal_zeta / np.pi * 180:.2f}')

probs_true = np.abs(np.sin(optimal_zeta)) ** 2
probs_false = np.abs(np.cos(optimal_zeta)) ** 2


shots = 10000
results = cudaq.sample(kernel_grover, qb, nums, iters, shots_count=shots)
# print(results)

result_final = np.zeros(2**qb)
probs = np.zeros_like(result_final)
pt = 0
for i in results:
    result_final[int(i, 2)] = results[i]
    probs[int(i, 2)] = results[i] / shots
    if int(i, 2) in nums:
        pt += probs[int(i, 2)]
# print(probs)
print(f'Probability of measuring True: {pt:.5f}')
print(f'Probability of Theoretical True: {probs_true:.5f}')

print(f'\nProbability of measuring False: {1-pt:.5f}')
print(f'Probability of Theoretical False: {probs_false:.5f}')