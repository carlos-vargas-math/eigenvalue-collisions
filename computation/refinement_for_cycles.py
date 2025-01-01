import numpy as np
import random_matrix_model.points_on_curve as curves
import computation.random_matrix_model as rmmodel
from permutation_utils import find_permutation
import datatypes
import datatypes1

def insert_refinement_points_for_cycles(summary):
    initial_matrix_data = summary['initial_matrix']
    initial_matrix = initial_matrix_data['matrix']
    curve = initial_matrix_data['curve']
    data = summary['summary_items']
    number_of_items = len(data)
    dim = initial_matrix.shape[0]

    # Find the indices where consecutive permutations differ by a longer cycle
    insert_points = []
    for i in range(10, number_of_items -1):
        permutation_indices = data['associated_permutation'][i]
        permutation_indices_next = data['associated_permutation'][i + 1]
        difference_permutation = find_permutation.find_resultant_permutation(permutation_indices, permutation_indices_next)
        cycle_decomposition = find_permutation.cycle_decomposition(difference_permutation)
        if find_permutation.has_longer_cycles(cycle_decomposition):
            insert_points.append(i)
            data['ordered'][i] = False

    custom_dtype = datatypes.create_custom_dtype(dim)

    # Initialize the updated data array with extra space for midpoint insertions
    updated_data = np.zeros(number_of_items + len(insert_points), custom_dtype)

    counter = 0  # This tracks how many items we've inserted
    aux_prev_i = 0  # This tracks the last position we copied from

    # Loop through each insert point and insert a midpoint
    for i in insert_points:
        print("adding new point at step" + str(i))
        # Copy everything from the previous point up to the current point
        updated_data[aux_prev_i + counter:i + counter] = data[aux_prev_i:i]

        # Calculate the midpoints for t and s
        t_0 = data["t"][i-1]
        t_1 = data["t"][i]
        t_new = (t_0 + t_1) / 2
        print("intermediate t =" + str(t_new))

        s_0 = data["s"][i-1]
        s_1 = data["s"][i]
        s_new = (s_0 + s_1) / 2
        print("intermediate s =" + str(s_new))

        # Insert the midpoint at position i + counter
        current_item = i + counter
        updated_data[current_item]['t'] = t_new
        updated_data[current_item]['s'] = s_new
        updated_data[current_item]['eigenvalues'] = np.linalg.eigvals(rmmodel.simple_flush_ginibre_toilet(initial_matrix, s_new, t_new, curve))  # Complex eigenvalues
        updated_data[current_item]['ordered'] = False

        # Update the counters
        counter += 1
        aux_prev_i = i

    # Copy the remaining part of loaded_item after the last insertion point
    updated_data[aux_prev_i + counter:] = data[aux_prev_i:]
    summary['summary_items'] = updated_data
    return summary
