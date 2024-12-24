import numpy as np
from scipy.spatial import Delaunay

# For eigenvalues of consecutive steps of R(t_0,s_0), R(t_1,s_1),
# Computes permutation obtained by considering the smallest distances.
# If this is not a correspondence the method fails 
# (and intermediate steps in [t_0,t_1], [s_0, s_1] should be considered)  
# Returns tripple [w_reordered, success (boolean), permutation].
def get_smallest_permutation_delaunay(z, w):
    points_with_favorite_neighbor = set()
    favorite_neighbors = set()
    n = z.size    
    points_z = np.array([[z[i].real, z[i].imag] for i in range(n)])
    points_w = np.array([[w[i].real, w[i].imag] for i in range(n)])
    points_combined = np.concatenate((points_z, points_w), axis=0) 

    # The function Delaunay returns a list of triples of indices (each triplet is a Delaunay triangle) 
    triangles = Delaunay(points_combined)
    k = len(triangles.simplices)

    current_index_map = {}
    result = np.zeros(n, dtype=complex)
    permutation = np.zeros(n, dtype=int)  # Track permutation
    
    for i in range(k):
        sorted_triplet = np.sort([triangles.simplices[i][0], triangles.simplices[i][1], triangles.simplices[i][2]])
        for pair in get_bipartite_edges_2(n, sorted_triplet[0], sorted_triplet[1], sorted_triplet[2]):
            if pair[0] in current_index_map:
                t_0 = z[pair[0]]
                t_1 = w[pair[1] - n]
                t_2 = result[pair[0]]  # This should refer to result, not w
                if np.abs(t_1 - t_0) < np.abs(t_2 - t_0):
                    result[pair[0]] = t_1  # Store reordered eigenvalue from w
                    permutation[pair[0]] = pair[1] - n  # Store the permutation
                    favorite_neighbors.add(pair[1])                
                    current_index_map.update({pair[0]: pair[1]})
            else:
                result[pair[0]] = w[pair[1] - n]  # Store eigenvalue from w
                permutation[pair[0]] = pair[1] - n  # Store the permutation
                points_with_favorite_neighbor.add(pair[0])
                favorite_neighbors.add(pair[1])
                current_index_map.update({pair[0]: pair[1]})   

    # Resolve conflicts
    codomain_set = set()
    complement = set(range(n))
    conflicting_pairs = set()
    
    for key in current_index_map:
        adjusted_index = current_index_map[key] - n
        if adjusted_index in codomain_set:
            conflicting_pairs.add(key)
        else:
            codomain_set.add(adjusted_index)
        complement.discard(adjusted_index)        

    # Check if we have a full bijection, if not, handle conflicts
    if len(current_index_map) != n or len(codomain_set) != n:
        for key in complement:
            for element in conflicting_pairs:
                if np.abs(z[element] - w[key]) < np.abs(z[element] - w[current_index_map[element] - n]):
                    current_index_map.update({element: key})
                    permutation[element] = key  # Update permutation as well

    if len(current_index_map) != n or len(codomain_set) != n:
        print("failed to obtain bijective solution")
        for key in complement:
            print(key)
            print(w[key])
        return [w, 0, np.arange(n)]  # Return identity permutation if failure

    # Return both the permuted eigenvalues and the permutation
    return [result, 1, permutation]

def get_bipartite_edges_2(n, i, j, k):
    result = []
    if i < n and j > n - 1:
        result.append([i,j])
    if i < n and k > n - 1:
        result.append([i,k])
    if j < n and i > n - 1:
        result.append([j,i])
    if j < n and k > n - 1:
        result.append([j,k])
    if k < n and i > n - 1:
        result.append([k,i])
    if k < n and j > n - 1:
        result.append([k,j])
    return result



