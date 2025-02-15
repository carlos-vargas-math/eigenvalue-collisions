import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import unordered_s_increase_eigenvalue_writter as s_unordered
from computation import s_eigenvalue_orderer as s_orderer
import computation.unorderered_refinement as unorderered_refinement
from permutation_utils import find_permutation
import datatypes1
import random_matrix_model.initial_matrix_writter as initial_matrix_writter
from computation import main1
from computation_rect import unordered_linear_segment_eigenvalue_writter
import numpy as np
import math

def count_collisions_recursive(initial_matrix_type, s_0, s_1, t_0, t_1, curve, max_depth=10, depth=0):
    """
    Recursively count collisions in the (s, t) square using divide-and-conquer.
    
    Parameters:
        initial_matrix_type: Initial matrix data.
        s_0, s_1, t_0, t_1: Boundaries of the current square.
        curve: Curve type for matrix generation.
        max_depth: Maximum recursion depth to avoid infinite recursion.
        depth: Current recursion depth.
    
    Returns:
        Total number of collisions in this region.
    """
    if depth > max_depth:
        return 0  # Prevent infinite recursion
    
    # Dynamically compute m based on the square size
    m = math.ceil((s_1 - s_0) * 10000)
    
    # Compute eigenvalues along the square path
    rectangle_data = unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
        initial_matrix_type, s_0, t_0, s_1, t_0, m
    )
    rectangle_data = np.concatenate((
        rectangle_data,
        unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
            initial_matrix_type, s_1, t_0, s_1, t_1, m
        )[1:]
    ))
    rectangle_data = np.concatenate((
        rectangle_data,
        unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
            initial_matrix_type, s_1, t_1, s_0, t_1, m
        )[1:]
    ))
    rectangle_data = np.concatenate((
        rectangle_data,
        unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
            initial_matrix_type, s_0, t_1, s_0, t_0, m
        )[1:]
    ))
    
    # Order eigenvalues and check for unordered steps
    rectangle_data, unordered_steps = s_orderer.order_s_eigenvalues(rectangle_data)
    while unordered_steps > 0:
        rectangle_data = unorderered_refinement.insert_unordered_refinement_points(
            initial_matrix_type['matrix'], rectangle_data, curve
        )
        rectangle_data, unordered_steps = s_orderer.order_s_eigenvalues(rectangle_data)
    
    # Analyze eigenvalue permutation cycles
    eigenvalues = rectangle_data['eigenvalues']
    z = eigenvalues[0, :]
    w = eigenvalues[-1, :]
    permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
    cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
    
    # Check for longer cycles
    if depth == 0 or find_permutation.has_longer_cycles(cycle_decomposition):
        # Subdivide the square into 4 subregions
        mid_s = (s_0 + s_1) / 2
        mid_t = (t_0 + t_1) / 2
        
        return (
            count_collisions_recursive(initial_matrix_type, s_0, mid_s, t_0, mid_t, curve, max_depth, depth + 1) +
            count_collisions_recursive(initial_matrix_type, mid_s, s_1, t_0, mid_t, curve, max_depth, depth + 1) +
            count_collisions_recursive(initial_matrix_type, s_0, mid_s, mid_t, t_1, curve, max_depth, depth + 1) +
            count_collisions_recursive(initial_matrix_type, mid_s, s_1, mid_t, t_1, curve, max_depth, depth + 1)
        )
    else:
        simplified_permutation = find_permutation.omit_singletons(permutation_indices)
        return find_permutation.cycle_length_sum(simplified_permutation)


# define parameter values
dim = 40
distribution = 'complexGaussian'
remove_trace = True
curve = curves.Curve.CIRCLE
seed = 1000

compute_summary = True
summary_name = "N=" + str(dim) + "&" + str(curve) + "&Seed=" + str(seed) + "&Distribution=" + distribution + "&Traceless=" + str(remove_trace)

initial_matrix_type = initial_matrix_writter.generate_initial_matrix(dim, distribution, remove_trace, seed, curve)
initial_matrix = initial_matrix_type['matrix']

# Example usage
number_of_collisions = count_collisions_recursive(
    initial_matrix_type, s_0=0.001, s_1=1, t_0=0, t_1=1, curve=curve
)
print("Total number of collisions:", number_of_collisions)



