import numpy as np
from computation import unordered_s_increase_eigenvalue_writter as s_unordered
from computation import s_eigenvalue_orderer as s_orderer
import computation.unorderered_refinement as unorderered_refinement
from computation import unordered_t_increase_eigenvalue_writter as t_unordered
from computation import t_eigenvalue_orderer as t_ordered
import permutation_utils.find_permutation as find_permutation


def compute_t_data(s_time_step, initial_t_steps, initial_matrix, s_data, curve):
    print(s_time_step)
    t_data = t_unordered.get_unordered_t_eigenvalues(initial_matrix, s_data, s_time_step, initial_t_steps, curve)
    t_data, unordered_steps = t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
    while unordered_steps > 0:
        t_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, t_data, curve)
        t_data, unordered_steps = t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
    return t_data

def compute_t_data_and_save_summary(s_time_step, initial_t_steps, summary, summary_name):
    initial_matrix_data = summary['initial_matrix']
    initial_matrix = initial_matrix_data['matrix']
    curve = initial_matrix_data['curve']
    s_data = summary['summary_items']

    print(s_time_step)
    t_data = t_unordered.get_unordered_t_eigenvalues(initial_matrix, s_data, s_time_step, initial_t_steps, curve)
    t_data, unordered_steps = t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
    while unordered_steps > 0:
        t_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, t_data, curve)
        t_data, unordered_steps = t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
        eigenvalues = t_data['eigenvalues']  # This is a (N_matrices, N_eigenvalues) array of complex eigenvalues
        # np.save('ordered_t_eigenvalues.npy', t_data)

        z = eigenvalues[0, :]
        w = eigenvalues[-1, :]
        permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
        cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
        print(cycle_decomposition)

        s_data['associated_permutation'][s_time_step] = permutation_indices
        summary['summary_items'] = s_data

    np.save(summary_name, summary)

    # optional save t_data

    return summary


def comptute_s_data(initial_matrix, initial_s_steps, curve):
    s_data = s_unordered.get_unordered_s_increasing_eigenvalues(initial_matrix, initial_s_steps, curve)
    s_data, unordered_steps = s_orderer.order_s_eigenvalues(s_data)
    while unordered_steps > 0:
        s_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, curve)
        s_data, unordered_steps = s_orderer.order_s_eigenvalues(s_data)
    return s_data
