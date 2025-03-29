import networkx as nx
import matplotlib.pyplot as plt
import random

def draw_graph(self):
    """Draws the graph with nodes, edges, and edge weights."""
    pos = nx.spring_layout(self)
    nx.draw(self, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(self, 'weight')
    nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_size=10)
    plt.show()

def draw_partition_graph(self, bitstring: list[int]):
    """Draws a bipartite view of the graph based on a bitstring partition."""
    nodes = list(self.nodes)
    if len(bitstring) != len(nodes):
        raise ValueError("Bitstring length must match the number of nodes.")
    
    S = [nodes[i] for i in range(len(nodes)) if bitstring[i] == 0]
    T = [nodes[i] for i in range(len(nodes)) if bitstring[i] == 1]

    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(S, bipartite=0)
    bipartite_graph.add_nodes_from(T, bipartite=1)

    cut_edges = []
    non_cut_edges = []
    for u, v, data in self.edges(data=True):
        if (u in S and v in T) or (u in T and v in S):
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))
    
    bipartite_graph.add_edges_from(cut_edges)
    bipartite_graph.add_edges_from(non_cut_edges)

    pos = nx.drawing.layout.bipartite_layout(bipartite_graph, S)
    nx.draw(bipartite_graph, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10)
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=cut_edges,
                           edge_color='red', width=2, label="Cut Edges")
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=non_cut_edges,
                           edge_color='gray', style="dashed", label="Non-Cut Edges")
    edge_labels = nx.get_edge_attributes(self, 'weight')
    nx.draw_networkx_edge_labels(bipartite_graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Bipartite Graph Representation for MaxCut with Highlighted Edges")
    plt.show()

@classmethod
def get_random_graph(cls, n, prob=0.5, weighted=True, seed=None):
    """Creates and returns a random graph with n nodes."""
    G = cls()
    V = range(n)
    G.add_nodes_from(V)
    if seed is not None:
        random.seed(seed)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() > prob:
                w = int(random.uniform(1, 10)) if weighted else 1
                G.add_edge(i, j, weight=w)
    return G

@classmethod
def get_graph_from_path(cls, path):
    """Reads a graph from a file."""
    G = cls()
    with open(path, 'r') as file:
        first_line = file.readline().strip()
        num_nodes, num_edges = map(int, first_line.split())
        for line in file:
            if line.strip():
                node1, node2, weight = line.strip().split()
                G.add_edge(int(node1), int(node2), weight=float(weight))
    return G

@classmethod
def evaluate_cut(self, bitstring: list[int], Graphdata, is_weighted: bool = True) -> float:
    """
    Evaluates the cut value of the graph for a given partition represented by a bitstring.
    
    Parameters:
        bitstring (list[int]): A list of integers (0 or 1) representing the partition of nodes.

        Graphdata (list[tuple]): A list of tuples representing the edges of the graph.

        is_weighted (bool): Flag to determine if the graph is weighted.
        
    Returns:
        float: The sum of weights for edges crossing the partition.
        
    Raises:
        ValueError: If the length of the bitstring does not match the number of nodes.
    """ 
    cut_value = 0
    for u, v, data in Graphdata:
        if bitstring[u] != bitstring[v]:
            cut_value += data.get('weight', 1) if is_weighted else 1
    return cut_value



# Monkey-patch the nx.Graph class
nx.Graph.draw_graph = draw_graph
nx.Graph.draw_partition_graph = draw_partition_graph
nx.Graph.get_random_graph = get_random_graph
nx.Graph.get_graph_from_path = get_graph_from_path
nx.Graph.evaluate_cut = evaluate_cut