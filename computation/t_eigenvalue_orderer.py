import numpy as np
import random_matrix_model.points_on_curve as curves
import computation.random_matrix_model as rmmodel
import permutation_utils.minimum_bipartite_matching as minimum_bipartite_matching
import datatypes1

def order_t_increasing_eigenvalues(data, s_data, s_step):
    dim = len(data['eigenvalues'][0])
    number_of_items = len(data)
    print(number_of_items)
    custom_dtype = datatypes1.create_summary_item_dtype(dim)

    # Initialize the updated data array with zeros
    updated_data = np.zeros(number_of_items, custom_dtype)

    # Initialize z with the eigenvalues of the first matrix
    z = s_data['eigenvalues'][s_step]

    # Copy the first entry of loaded_item to updated_data (since there's no need to reorder the first)
    updated_data[0] = s_data[s_step]

    # Loop over the remaining matrices to reorder the eigenvalues
    i = 1
    unordered_steps = 0
    while i < number_of_items:
        
        # Copy all other fields from loaded_item to updated_data for this index
        updated_data[i] = data[i]

        # Get the next eigenvalues to reorder
        w = data['eigenvalues'][i]
        
        # Computes the rearrangement of w that minimizes 
        new_w, ordered, permutation = minimum_bipartite_matching.delaunay_bipartite_matching(z, w)

        # Update the 'eigenvalues' field in the updated_data array
        updated_data[i]['eigenvalues'] = new_w
        
        # Set 'ordered' to True since we reordered this entry
        if ordered == 0:
            print("failed to obtain bijective solution at step " + str(i))
            print("interval t: " + str(data['t'][i-1]) + " to " + str(data['t'][i]))
            # Since there's an issue, continue to the next index
            i += 1
            unordered_steps += 1
        else:
            updated_data[i]['ordered'] = True
            # Check consecutive entries
            i += 1
            while i < number_of_items and updated_data[i]['ordered']: 
                # If already ordered, apply the previous permutation
                updated_data[i]['eigenvalues'] = permutation @ data['eigenvalues'][i]
                updated_data[i]['ordered'] = True
                i += 1
        
        # Update z for the next iteration
        z = new_w
    return [updated_data , unordered_steps]
