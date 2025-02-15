import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import unordered_s_increase_eigenvalue_writter as s_unordered
from computation import s_eigenvalue_orderer as s_orderer
import computation.unorderered_refinement as unorderered_refinement
from permutation_utils import find_permutation
import datatypes1
import random_matrix_model.initial_matrix_writter as initial_matrix_writter
from computation import main1

# set initial matrix and settings

summary_name = "computed_examples/N=20&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True.npy"
summary = np.load(summary_name, allow_pickle=True)
s_data = summary['summary_items']

if (compute_summary == True):
    print("actual_s_steps" + str(actual_s_steps))
    for s_time_step in range(1 , actual_s_steps): 

        t_data = main1.compute_t_data(s_time_step, initial_t_steps, initial_matrix, s_data, curve)
        eigenvalues = t_data['eigenvalues']  # This is a (N_matrices, N_eigenvalues) array of complex eigenvalues
        # np.save('ordered_t_eigenvalues.npy', t_data)

        z = eigenvalues[0, :]
        w = eigenvalues[-1, :]
        permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
        cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
        print(cycle_decomposition)

        s_data['associated_permutation'][s_time_step] = permutation_indices

    np.save('ordered_s_eigenvalues.npy', s_data)
    ginibre_summary['summary_items'] = s_data

    np.save(summary_name, ginibre_summary)

