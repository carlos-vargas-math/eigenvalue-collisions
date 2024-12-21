import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import unordered_s_increase_eigenvalue_writter as s_unordered
from computation import s_eigenvalue_orderer as s_orderer
import computation.unorderered_refinement as unorderered_refinement
from computation import unordered_t_increase_eigenvalue_writter as t_unordered
from computation import t_eigenvalue_orderer as t_ordered
import permutation_utils.find_permutation as find_permutation


def compute_t_data(s_time_step, initial_t_steps, initial_matrix, s_data, curve):
    print(s_time_step)
    t_data = t_unordered.get_unordered_t_eigenvalues(initial_matrix, s_data, s_time_step, initial_t_steps)
    t_data, unordered_steps = t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
    while unordered_steps > 0:
        t_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, t_data, curve)
        t_data, unordered_steps = t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
    return t_data

# Not recommended... as it is it may skip some collisions.
def divide_and_conquer_eigenvalues(
    s_data, initial_matrix, curve, s_0, s_1, initial_t_steps, initial_s_steps,
    t_unordered, t_ordered, unordered_refinement, visited_steps
):
    """
    Compute eigenvalues and permutation indices using divide and conquer.
    Ensure each step is visited only once.
    """
    # Check if the interval [s_0, s_1] has already been processed
    if s_0 in visited_steps and s_1 in visited_steps:
        return

    def compute_t_data(s_time_step, initial_t_steps):
        print(s_time_step)
        t_data = t_unordered.get_unordered_t_eigenvalues(initial_matrix, s_data, s_time_step, initial_t_steps)
        t_data, unordered_steps = t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
        while unordered_steps > 0:
            t_data = unordered_refinement.insert_unordered_refinement_points(initial_matrix, t_data, curve)
            t_data, unordered_steps = t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
        return t_data

    # Compute for s_0
    if s_0 not in visited_steps:
        t_data_0 = compute_t_data(s_0, initial_t_steps)
        eigenvalues_0 = t_data_0['eigenvalues']
        z_0 = eigenvalues_0[0, :]
        w_0 = eigenvalues_0[-1, :]
        _, permutation_indices_0 = find_permutation.find_best_permutation(z_0, w_0)
        s_data['associated_permutation'][s_0] = permutation_indices_0
        visited_steps.add(s_0)
    else:
        permutation_indices_0 = s_data['associated_permutation'][s_0]

    # Compute for s_1
    if s_1 not in visited_steps:
        t_data_1 = compute_t_data(s_1, initial_t_steps)
        eigenvalues_1 = t_data_1['eigenvalues']
        z_1 = eigenvalues_1[0, :]
        w_1 = eigenvalues_1[-1, :]
        _, permutation_indices_1 = find_permutation.find_best_permutation(z_1, w_1)
        s_data['associated_permutation'][s_1] = permutation_indices_1
        visited_steps.add(s_1)
    else:
        permutation_indices_1 = s_data['associated_permutation'][s_1]

    # Optimization: Skip interval if permutations match, except for [10, 10000]
    if np.array_equal(permutation_indices_0, permutation_indices_1) and not (s_0 == 10 and s_1 == initial_s_steps):
        print('filling up all permutations in interval' + str([s_0, s_1]))
        s_data['associated_permutation'][s_0:s_1 + 1] = permutation_indices_0
        return

    # Recurse for the midpoint
    s_m = (s_0 + s_1) // 2
    if s_m not in visited_steps:
        divide_and_conquer_eigenvalues(
            s_data, initial_matrix, curve, s_0, s_m, initial_t_steps, initial_s_steps,
            t_unordered, t_ordered, unordered_refinement, visited_steps
        )
        divide_and_conquer_eigenvalues(
            s_data, initial_matrix, curve, s_m + 1, s_1, initial_t_steps, initial_s_steps,
            t_unordered, t_ordered, unordered_refinement, visited_steps
        )
