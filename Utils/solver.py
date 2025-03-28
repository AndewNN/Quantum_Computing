from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridSampler
from amplify import DWaveSamplerClient
from amplify import FixstarsClient, solve, VariableGenerator
from datetime import timedelta
from gurobi_optimods.qubo import solve_qubo
from qiskit_optimization.applications import Maxcut
import networkx as nx
import graph as ut
from dotenv import load_dotenv
import os
import time

load_dotenv()
FIXSTAR_TOKEN = os.getenv("FIXSTAR_TOKEN")
DWAVE_TOKEN  = os.getenv("DWAVE_TOKEN")

# Construct Quadratic and Linear part and construct module
def getModule(G: nx.Graph):
    # Return m: Module
    w = nx.adjacency_matrix(G).todense()
    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()
    Quadratic = qp.objective.quadratic.to_array()
    Linear = qp.objective.linear.to_array()

    gen = VariableGenerator()
    m = gen.matrix("Binary", len(G.nodes()))
    m.quadratic = -Quadratic
    m.linear = -Linear
    return m

# Gurobi
def solveWithGurobi(G: nx.Graph):
    Q = ut.get_QUBO_Matrix(G)

    start_time = time.time()
    result = solve_qubo(-Q, solver_params={'MIPGap': 0.01})
    end_time = time.time()
    takeTime = end_time - start_time
    takeTime = timedelta(seconds=takeTime)

    ret = {
        'execution_time': takeTime,
        'best_values': result.solution,
        'objective': result.objective_value,
    }
    return ret

# Fixstar
def solveWithFixstar(G: nx.Graph):
    fsClient = FixstarsClient()
    fsClient.token = FIXSTAR_TOKEN
    fsClient.parameters.timeout = timedelta(milliseconds=1000)
    fsClient.parameters.outputs.num_outputs = 0

    m = getModule(G)
    result = solve(m, fsClient)
    arr = m.variable_array

    ret = {
        'execution_time': result.execution_time,
        'best_values': arr.evaluate(result.best.values),
        'objective': result.best.objective,
    }
    return ret

# DWave
def solveWithDWave(G: nx.Graph, solver="Advantage_system6.4", num_reads=1000):
    dwClient = DWaveSamplerClient()
    dwClient.token = DWAVE_TOKEN
    dwClient.solver = solver
    dwClient.parameters.num_reads = num_reads

    m = getModule(G)
    result = solve(m, dwClient)

    ret = {
        'execution_time': result.execution_time,
        'best_values': result.best.values,
        'objective': result.best.objective,
    }
    return ret

def solveWithHybridDwave(G: nx.Graph) :
    sampler = LeapHybridSampler(token=DWAVE_TOKEN)
    
    Q = ut.get_QUBO_Matrix(G)
    response = sampler.sample_qubo(-Q)

    solution  = [response.first.sample[b] for b in response.first.sample]

    ret = {
        'execution_time': response.info['qpu_access_time']/10**6,
        'best_value': solution,
        'objective': response.first.energy
    }
    return ret