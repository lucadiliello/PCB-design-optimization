from geometry import Graph
import random

# generate a random graph with n_nodes nodes and n_edges edges.
# Each node will have coordinates in ranges.
def generate_circuit(n_nodes, n_edges, range=(0,50)):
    if n_edges > n_nodes * (n_nodes-1)/2:
        # this graph cannot exists
        return None

    nodes = [(random.uniform(*range), random.uniform(*range)) for i in range(n_nodes)]

    all_edges = list(itertools.combinations(range(n_nodes), 2))
    edges = []
    for i in range(n_edges):
        j = random.randint(0,len(all_edges)-1)
        edges.append(all_edges[j])
        del all_edges[j]

    return Graph(nodes, edges)
