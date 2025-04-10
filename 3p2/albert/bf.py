import numpy as np
import math

def bellman_ford(graph, src):
    n = len(graph)
    dist = [float('inf')] * n
    pred = [-1] * n
    dist[src] = 0

    # Relaxation step: update distances and predecessors
    for _ in range(n - 1):
        for u in range(n):
            for v in range(n):
                if dist[u] + graph[u][v] < dist[v]:
                    dist[v] = dist[u] + graph[u][v]
                    pred[v] = u

    print(dist)
    print(pred)

    # Check for negative weight cycle:
    # If any edge can be relaxed, a negative cycle exists.
    cycle_vertex = None
    for u in range(n):
        for v in range(n):
            if dist[u] + graph[u][v] < dist[v]:
                cycle_vertex = v
                break
        if cycle_vertex is not None:
            break

    if cycle_vertex is None:
        return None  # No profitable (negative) cycle found
    else:
        return find_cycle(pred, cycle_vertex)

def find_cycle(pred, start):
    """
    Reconstructs a cycle using the predecessor array.
    We first "backtrack" n times from the starting vertex to ensure
    that we are inside the cycle, then build the cycle by following the predecessor chain.
    """
    n = len(pred)
    cycle_vertex = start
    # Backtrack n times to ensure the vertex is on a cycle.
    for _ in range(n):
        cycle_vertex = pred[cycle_vertex]

    # Now, starting at cycle_vertex, reconstruct the cycle.
    cycle = []
    current = cycle_vertex
    while True:
        cycle.append(current)
        current = pred[current]
        if current == cycle_vertex:
            cycle.append(current)  # to close the cycle
            break

    # Reverse the cycle to show it in the actual order of transactions.
    cycle.reverse()
    return cycle

# Exchange rates matrix where exchange_rates[i][j] is the rate from currencies[i] to currencies[j]
exchange_rates = [
    [1,    1.45, 0.52, 0.72],  # Snowballs to Snowballs, Pizzas, Silicon Nuggets, SeaShells
    [0.7,  1,    0.31, 0.48],  # Pizzas
    [1.95, 3.1,  1,    1.49],  # Silicon Nuggets
    [1.34, 1.98, 0.64, 1]      # SeaShells
]

currencies = ['Snowballs', 'Pizzas', 'Silicon Nuggets', 'SeaShells']
n = len(currencies)
# Convert rates to a graph with weights as negative logarithms of the exchange rates.
graph = [[-np.log(rate) for rate in row] for row in exchange_rates]

# Choose a source vertex. It does not have to be in the cycle, but it must be able to reach the cycle.
source_index = 3  # for 'SeaShells'
profitable_cycle = bellman_ford(graph, source_index)

# Output the result
if profitable_cycle:
    # The cycle is represented as indices. Format it to show the currency names.
    print("Profitable cycle found:", [currencies[i] for i in profitable_cycle])
else:
    print("No profitable cycle found.")
