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

# define parameter values
dim = 10
distribution = 'complexGaussian'
remove_trace = True
curve = curves.Curve.CIRCLE
seed = 1008

compute_summary = False
summary_name = "N=" + str(dim) + "&" + str(curve) + "&Seed=" + str(seed) + "&Distribution=" + distribution + "&Traceless=" + str(remove_trace)

initial_matrix_type = initial_matrix_writter.generate_initial_matrix(dim, distribution, remove_trace, seed, curve)
initial_matrix = initial_matrix_type['matrix']

# Define the number of initial summary steps and initial rotation steps
initial_s_steps = 1000
initial_t_steps = 1000

s_data = main1.comptute_s_data(initial_matrix, initial_s_steps, curve)

np.save('ordered_s_eigenvalues.npy', s_data)

# Create a structured array instance using the Ginibre summary dtype
actual_s_steps = s_data.size
ginibre_summary_dtype = datatypes1.create_ginibre_summary_dtype(dim, actual_s_steps)
ginibre_summary = np.zeros((), dtype=ginibre_summary_dtype)

# Populate the fields
ginibre_summary['initial_matrix'] = initial_matrix_type

# Populate the Ginibre summary and save
ginibre_summary['summary_items'] = s_data
np.save(summary_name, ginibre_summary)

# compute t_data (this method takes quite long, 
# as it computes t_data for all points in the s_partition)
# If one wants to compute t_data for just a few s_steps, use main_t_data.py

# non-D&C method (D&C might be a bad idea)
last_permutation = np.arange(dim)
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
        difference_permutation = find_permutation.find_resultant_permutation(last_permutation, permutation_indices)
        if find_permutation.has_longer_cycles(difference_permutation):
            print("Will insert extra step, difference permutation has longer cycles")
            print(difference_permutation)
            # s_data['ordered'][s_time_step] = False
        last_permutation = permutation_indices

        print(cycle_decomposition)
        s_data['associated_permutation'][s_time_step] = permutation_indices


    np.save('ordered_s_eigenvalues.npy', s_data)
    ginibre_summary['summary_items'] = s_data

    np.save(summary_name, ginibre_summary)

    # Update the size of s_data if refinement has been applied
    actual_s_steps = len(s_data)

    # Recreate ginibre_summary with updated dtype
    ginibre_summary_dtype = datatypes1.create_ginibre_summary_dtype(dim, actual_s_steps)
    ginibre_summary = np.zeros((), dtype=ginibre_summary_dtype)

    # Assign the updated s_data
    ginibre_summary['summary_items'] = s_data
    # Populate the fields
    ginibre_summary['initial_matrix'] = initial_matrix_type

    # Save ginibre_summary
    np.save(summary_name, ginibre_summary)

    s_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, s_data, curve)
    s_data, unordered_steps = s_orderer.order_s_eigenvalues(s_data)

    actual_s_steps = len(s_data)
    print(actual_s_steps)
    for s_time_step in range(1 , actual_s_steps):        
        if np.array_equal(s_data['associated_permutation'][s_time_step], np.zeros(dim)):
            t_data = main1.compute_t_data(s_time_step, initial_t_steps, initial_matrix, s_data, curve)
            eigenvalues = t_data['eigenvalues']  # This is a (N_matrices, N_eigenvalues) array of complex eigenvalues
            # np.save('ordered_t_eigenvalues.npy', t_data)
            z = eigenvalues[0, :]
            w = eigenvalues[-1, :]
            permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
            cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
            s_data['associated_permutation'][s_time_step] = permutation_indices
            print(cycle_decomposition)
    
    np.save('refined_ordered_s_eigenvalues.npy', s_data)
    actual_s_steps = len(s_data)

    # Recreate ginibre_summary with updated dtype
    ginibre_summary_dtype = datatypes1.create_ginibre_summary_dtype(dim, actual_s_steps)
    ginibre_summary = np.zeros((), dtype=ginibre_summary_dtype)

    # Assign the updated s_data
    ginibre_summary['summary_items'] = s_data
    # Populate the fields
    ginibre_summary['initial_matrix'] = initial_matrix_type

    # Save ginibre_summary
    np.save(summary_name, ginibre_summary)

# current_repetition = 0
# number_of_final_refinements = 10
# while (current_repetition < number_of_final_refinements):
#     print("refinement number ", current_repetition)
#     current_repetition += 1
#     last_permutation = np.arange(dim)
#     print("actual_s_steps" + str(actual_s_steps))
#     for s_time_step in range(1 , actual_s_steps): 
#         permutation_indices = s_data['associated_permutation'][s_time_step]
#         difference_permutation = find_permutation.find_resultant_permutation(last_permutation, permutation_indices)
#         if find_permutation.has_longer_cycles(difference_permutation):
#             print("Will insert extra step, difference permutation has longer cycles")
#             print(difference_permutation)
#             s_data['ordered'][s_time_step] = False
#         last_permutation = permutation_indices

#     np.save('ordered_s_eigenvalues.npy', s_data)
#     ginibre_summary['summary_items'] = s_data

#     np.save(summary_name, ginibre_summary)

#     # Update the size of s_data if refinement has been applied
#     actual_s_steps = len(s_data)

#     # Recreate ginibre_summary with updated dtype
#     ginibre_summary_dtype = datatypes1.create_ginibre_summary_dtype(dim, actual_s_steps)
#     ginibre_summary = np.zeros((), dtype=ginibre_summary_dtype)

#     # Assign the updated s_data
#     ginibre_summary['summary_items'] = s_data
#     # Populate the fields
#     ginibre_summary['initial_matrix'] = initial_matrix_type

#     # Save ginibre_summary
#     np.save(summary_name, ginibre_summary)

#     s_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, s_data, curve)
#     s_data, unordered_steps = s_orderer.order_s_eigenvalues(s_data)

#     actual_s_steps = len(s_data)
#     print(actual_s_steps)
#     for s_time_step in range(1 , actual_s_steps):        
#         if np.array_equal(s_data['associated_permutation'][s_time_step], np.zeros(dim)):
#             t_data = main1.compute_t_data(s_time_step, initial_t_steps, initial_matrix, s_data, curve)
#             eigenvalues = t_data['eigenvalues']  # This is a (N_matrices, N_eigenvalues) array of complex eigenvalues
#             # np.save('ordered_t_eigenvalues.npy', t_data)
#             z = eigenvalues[0, :]
#             w = eigenvalues[-1, :]
#             permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
#             cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
#             s_data['associated_permutation'][s_time_step] = permutation_indices
#             print(cycle_decomposition)
    
#     np.save('refined_ordered_s_eigenvalues.npy', s_data)
#     actual_s_steps = len(s_data)

#     # Recreate ginibre_summary with updated dtype
#     ginibre_summary_dtype = datatypes1.create_ginibre_summary_dtype(dim, actual_s_steps)
#     ginibre_summary = np.zeros((), dtype=ginibre_summary_dtype)

#     # Assign the updated s_data
#     ginibre_summary['summary_items'] = s_data
#     # Populate the fields
#     ginibre_summary['initial_matrix'] = initial_matrix_type

#     # Save ginibre_summary
#     np.save(summary_name, ginibre_summary)
