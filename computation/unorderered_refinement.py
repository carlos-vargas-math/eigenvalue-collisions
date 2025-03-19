import numpy as np
import random_matrix_model.points_on_curve as curves
import computation.random_matrix_model as rmmodel
import datatypes1

def insert_unordered_refinement_points(initial_matrix, data, curve):
    number_of_items = len(data)
    dim = initial_matrix.shape[0]

    # Find the indices where 'ordered' is False
    insert_points = []
    for i in range(1, number_of_items):
        if data['ordered'][i] == False:
            insert_points.append(i)

    custom_dtype = datatypes1.create_summary_item_dtype(dim)
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
    return updated_data
