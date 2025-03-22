import numpy as np
import random_matrix_model.points_on_curve as curves
import computation.random_matrix_model as rmmodel
import datatypes1

def get_unordered_t_eigenvalues(initial_matrix, s_data, s_step, number_of_t_steps, curve):

    # Load the single dtype item from the .npy file
    s_initial_eigenvalues = s_data[s_step]

    initial_t = 0
    final_t = 1
    initial_s = s_initial_eigenvalues['s']
    number_of_steps = number_of_t_steps
    number_of_items = number_of_steps + 1
    t_increase = (final_t - initial_t) / number_of_steps
    dim = initial_matrix.shape[0]
    print(dim)
    custom_dtype = datatypes1.create_summary_item_dtype(dim)

    # Create the structured array
    data = np.zeros(number_of_items, custom_dtype)
    data[0] = s_initial_eigenvalues
    data[0]['ordered'] = False
    
    t = initial_t
    s = initial_s
    if s < 0.00000001:
        data[0]['ordered'] = True

    # Fill the array with some sample data
    for i in range(number_of_steps):
        i += 1
        t += t_increase
        # print(i)
        data[i]['t'] = t
        data[i]['s'] = s
        if s > 0.00000001:
            data[i]['eigenvalues'] = np.linalg.eigvals(rmmodel.simple_generate_ginibre_whirlwind(initial_matrix, s, t, curve))  # Complex eigenvalues
            data[i]['ordered'] = False
        else:
            data[i]['eigenvalues'] = data[0]['eigenvalues']  # Complex eigenvalues
            data[i]['ordered'] = True

    # Save the structured array to a .npy file
    return data

