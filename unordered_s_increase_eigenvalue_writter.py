import numpy as np
import points_on_curve as curves
import random_matrix_model as rmmodel
import datatypes1

def get_unordered_s_increasing_eigenvalues(initial_matrix, number_of_steps, curve):
    initial_s = 0
    final_s = 1
    number_of_items = number_of_steps + 1
    s_increase = (final_s - initial_s) / number_of_steps 
    dim = initial_matrix.shape[0]

    custom_dtype = datatypes1.create_summary_item_dtype(dim)
    # Create the structured array
    data = np.zeros(number_of_items, custom_dtype)

    t = 0
    s = initial_s
    # Fill the array with some sample data
    for i in range(number_of_items):
        # print(i)
        data[i]['t'] = t
        data[i]['s'] = s
        data[i]['eigenvalues'] = np.linalg.eigvals(rmmodel.simple_flush_ginibre_toilet(initial_matrix, s, t, curve))  # Complex eigenvalues
        data[i]['ordered'] = False
        if i == 0:
            data[i]['ordered'] = True
        s += s_increase

    z = data[0]['eigenvalues']
    data[0]['eigenvalues'] = sorted(z, key=abs)

    return data