import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import unordered_s_increase_eigenvalue_writter as s_unordered
from computation import s_eigenvalue_orderer as s_orderer
import computation.unorderered_refinement as unorderered_refinement
from computation import unordered_t_increase_eigenvalue_writter as t_unordered
import well_ordered_summaries as t_ordered
import datatypes1
import random_matrix_model.initial_matrix_writter as initial_matrix_writter

# set initial matrix and settings

curve = curves.Curve.CIRCLE

# Generate initial matrix
dim = 70
distribution = 'complexGaussian'
remove_trace = True
initial_matrix_type = initial_matrix_writter.generate_initial_matrix(dim, distribution, remove_trace, seed=998)
initial_matrix = initial_matrix_type['matrix']

# Define the number of initial summary steps and initial rotation steps
initial_s_steps = 2000
initial_t_steps = 1000

s_data = s_unordered.get_unordered_s_increasing_eigenvalues(initial_matrix, initial_s_steps, curve)
s_data = s_orderer.order_s_eigenvalues(s_data)
s_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, s_data, curve)

np.save('ordered_s_eigenvalues.npy', s_data)

# Create a structured array instance using the Ginibre summary dtype
actual_s_steps = s_data.size
ginibre_summary_dtype = datatypes1.create_ginibre_summary_dtype(dim, actual_s_steps)
ginibre_summary = np.zeros((), dtype=ginibre_summary_dtype)

# Populate the fields
ginibre_summary['initial_matrix'] = initial_matrix_type

# Populate the Ginibre summary and save
ginibre_summary['summary_items'] = s_data
np.save('ginibre_summary_type.npy', ginibre_summary)

# # compute t_data (this method takes quite long, 
# as it computes t_data for al points in the s_partition)
# If one wants to show 

# non-D&C method (D&C might be a bad idea)
# for s_time_step in range(10,initial_s_steps): 
#     print(s_time_step)
#     t_data = t_unordered.get_unordered_t_eigenvalues(initial_matrix, s_data, s_time_step, initial_t_steps)
#     t_data , unordered_steps =  t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)
#     while (unordered_steps > 0):
#         print(unordered_steps)    
#         t_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, t_data, curve)
#         t_data , unordered_steps =  t_ordered.order_t_increasing_eigenvalues(t_data, s_data, s_time_step)

#     eigenvalues = t_data['eigenvalues']  # This is a (N_matrices, N_eigenvalues) array of complex eigenvalues
#     np.save('ordered_t_eigenvalues.npy', t_data)

#     z = eigenvalues[0, :]
#     w = eigenvalues[-1, :]
#     permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
#     cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
#     print(cycle_decomposition)

#     s_data['associated_permutation'][s_time_step] = permutation_indices

# np.save('ordered_s_eigenvalues.npy', s_data)
# ginibre_summary['summary_items'] = s_data

# np.save('ginibre_summary_type.npy', ginibre_summary)

# t_data = main1.compute_t_data(1000, initial_t_steps, initial_matrix, s_data, curve)

# np.save('ordered_t_eigenvalues', t_data)

