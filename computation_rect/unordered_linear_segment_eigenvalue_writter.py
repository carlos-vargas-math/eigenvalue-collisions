import numpy as np
import random_matrix_model.points_on_curve as curves
import computation.random_matrix_model as rmmodel
import datatypes1

# divides
def get_eigenvalues_along_segment(initial_matrix_type, s_0, t_0, s_1, t_1, m):
    initial_matrix = initial_matrix_type['matrix']
    curve = initial_matrix_type["curve"]
    initial_s = s_0
    final_s = s_1
    initial_t = t_0
    final_t = t_1

    number_of_items = m + 1
    s_increase = (final_s - initial_s) / m 
    t_increase = (final_t - initial_t) / m 

    dim = initial_matrix.shape[0]

    custom_dtype = datatypes1.create_summary_item_dtype(dim)
    # Create the structured array
    data = np.zeros(number_of_items, custom_dtype)

    t = initial_t
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
        t += t_increase

    # z = data[0]['eigenvalues']
    # data[0]['eigenvalues'] = sorted(z, key=abs)

    return data