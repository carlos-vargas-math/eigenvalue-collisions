import numpy as np
from scipy.optimize import linear_sum_assignment

def find_best_permutation(z, w):
    """
    z and w, are almost exaclty the same arrays of eigenvalues 
    (with differences due to rounding, etc), 
    possibly in different order. 
    The method returns the permuted elements of w and the required permutation. 
    """
    # Compute the distance matrix between z and w
    dist_matrix = np.abs(z[:, np.newaxis] - w[np.newaxis, :])

    # Use the Hungarian algorithm to find the optimal assignment (minimum cost matching)
    row_indices, col_indices = linear_sum_assignment(dist_matrix)

    # Apply the optimal permutation to w
    permuted_w = w[col_indices]

    return permuted_w, col_indices


def omit_singletons(cycles):
    return [cycle for cycle in cycles if len(cycle) > 1]

def cycle_decomposition(perm):
    n = len(perm)
    visited = [False] * n  # Track which elements have been visited
    cycles = []

    for i in range(n):
        if not visited[i]:
            # Start a new cycle
            current = i
            cycle = []
            
            # Follow the cycle
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = perm[current]
            
            # Add the cycle to the list of cycles
            cycles.append(cycle)
    
    return cycles


def cycle_length_sum(cycles):
    return sum(len(cycle) - 1 for cycle in cycles)


def find_resultant_permutation(perm1, perm2):
    """
    finds perm, such that perm2 = perm1 * perm
    """
    n = len(perm1)
    visited = [False] * n  # Track visited indices
    result_cycles = []

    for i in range(n):
        if visited[i]:
            continue

        if perm1[i] == perm2[i]:
            # Persistent element, forms a singleton
            result_cycles.append([i])
            visited[i] = True
        else:
            # Non-persistent element, form a cycle
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                # Transition to the next index via perm2
                next_value = perm2[current]
                current = list(perm1).index(next_value)
            result_cycles.append(cycle)

    return result_cycles
