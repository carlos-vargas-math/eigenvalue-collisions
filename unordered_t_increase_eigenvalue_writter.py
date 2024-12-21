import numpy as np
import points_on_curve as curves
import random_matrix_model as rmmodel
import datatypes
import datatypes1


# # Load the single dtype item from the .npy file
# loaded_item = np.load('initialMatrices/s_partition_50_2000.npy', allow_pickle=True)
# initial_matrix = loaded_item['initial_matrix'][0]
# matrix_uuid = loaded_item['initial_matrix_id'][0]

# s_step = 1000

# loaded_item_2 = np.load('ordered_s_eigenvalues.npy', allow_pickle=True)
# s_initial_eigenvalues = loaded_item_2[s_step]

# initial_t = 0
# final_t = 1
# # order_first_element_by_norm = True
# order_first_element_by_norm = False
# initial_s = s_initial_eigenvalues['s']
# final_s = s_initial_eigenvalues['s']
# number_of_steps = 2000
# number_of_items = number_of_steps + 1
# t_increase = (final_t - initial_t) / number_of_steps
# s_increase = (final_s - initial_s) / number_of_steps 
# dim = initial_matrix.shape[0]
# print(dim)
# curve = curves.Curve.CIRCLE

# # Accessing fields from the loaded data
# print("Matrix ID (UUID):", loaded_item['initial_matrix_id'])
# print("Initial matrix properties:", loaded_item['initial_matrix_properties'])
# print("Entries of the initial matrix:", loaded_item['initial_matrix'])
# print(np.linalg.eigvals(initial_matrix))

# # Define the dtype
# dtype = np.dtype([
#     ('t', np.float64),
#     ('s', np.float64),
#     ('initial_matrix_id', 'U36'),  # UUID stored as a string (36 characters including dashes)
#     ('problematic_indices', object),  # Use object to store lists of varying sizes
#     ('eigenvalues', np.complex128, (dim,)),  # Store N (1D) eigenvalues for each matrix
#     ('ordered', np.bool_),
#     ('permuted_eigenvalues', np.complex128, (dim,)),  # Store N (1D) eigenvalues for each matrix
# ])


# # Create the structured array
# data = np.zeros(number_of_items, dtype=dtype)
# data[0] = s_initial_eigenvalues
# data[0]['ordered'] = False

# t = initial_t
# s = initial_s
# # Fill the array with some sample data
# for i in range(number_of_steps):
#     i += 1
#     t += t_increase
#     s += s_increase
#     print(i)
#     data[i]['t'] = t
#     data[i]['s'] = s
#     data[i]['initial_matrix_id'] = str(matrix_uuid)
#     data[i]['problematic_indices'] = []  # Example problematic indices
#     data[i]['eigenvalues'] = np.linalg.eigvals(rmmodel.simple_flush_ginibre_toilet(initial_matrix, s, t, curve))  # Complex eigenvalues
#     data[i]['ordered'] = False


# if order_first_element_by_norm:
#     z = data[0]['eigenvalues']
#     data[0]['eigenvalues'] = sorted(z, key=abs)

# # Save the structured array to a .npy file
# np.save('unordered_t_eigenvalues.npy', data)

def get_unordered_t_eigenvalues(initial_matrix, s_data, s_step, number_of_t_steps):

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
    curve = curves.Curve.CIRCLE
    custom_dtype = datatypes1.create_summary_item_dtype(dim)

    # Create the structured array
    data = np.zeros(number_of_items, custom_dtype)
    data[0] = s_initial_eigenvalues
    data[0]['ordered'] = False

    t = initial_t
    s = initial_s
    # Fill the array with some sample data
    for i in range(number_of_steps):
        i += 1
        t += t_increase
        # print(i)
        data[i]['t'] = t
        data[i]['s'] = s
        data[i]['eigenvalues'] = np.linalg.eigvals(rmmodel.simple_flush_ginibre_toilet(initial_matrix, s, t, curve))  # Complex eigenvalues
        data[i]['ordered'] = False

    # Save the structured array to a .npy file
    return data

