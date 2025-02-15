import numpy as np
from scipy.spatial import Delaunay

def delaunay_bipartite_matching(z, w):
    n = len(z)
    z_points = np.column_stack((z.real, z.imag))
    w_points = np.column_stack((w.real, w.imag))

    # Combine points for triangulation
    all_points = np.vstack((z_points, w_points))
    delaunay = Delaunay(all_points)

    # Extract edges from Delaunay triangulation
    edges = set()
    for simplex in delaunay.simplices:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            if simplex[i] < n and simplex[j] >= n:
                edges.add((simplex[i], simplex[j] - n))
            elif simplex[j] < n and simplex[i] >= n:
                edges.add((simplex[j], simplex[i] - n))

    # Sort edges by distance
    edges = sorted(edges, key=lambda e: np.abs(z[e[0]] - w[e[1]]))

    # Greedy matching
    matched_z = set()
    matched_w = set()
    result = np.zeros(n, dtype=complex)
    permutation = np.zeros(n, dtype=int)

    for i, j in edges:
        if i not in matched_z and j not in matched_w:
            result[i] = w[j]
            permutation[i] = j
            matched_z.add(i)
            matched_w.add(j)

    # Check if we have a full matching
    success = len(matched_z) == n and len(matched_w) == n

    return [result, int(success), permutation]
