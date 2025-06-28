import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import s_eigenvalue_orderer as s_orderer
import computation.unorderered_refinement as unorderered_refinement
from permutation_utils import find_permutation
import random_matrix_model.initial_matrix_writter as initial_matrix_writter
from computation_rect import unordered_linear_segment_eigenvalue_writter
import time
from settings import settings, generate_directory_name
import os

# For statistics of multiple seed values, set load_parameters_from_settings = False
start_time = time.time()
seed_start = settings.seed
seed_end = settings.seed_end
seed_list = range(seed_start, seed_end + 1)
grid_value = settings.grid_m
grid_values = [grid_value]
dim = settings.dim
distribution = settings.distribution
remove_trace = settings.remove_trace
curve = settings.curve
if seed_end == seed_start:
    grid_summary_name = generate_directory_name() + "/gridm=" + str(grid_value) + ".npy"
else:
    grid_summary_name = "computed_examples/grid_summaries/N=" + str(dim) +"seedFrom" + str(seed_start) + "To" + str(seed_end)  + "&" + str(curve) + "&" + str(distribution) + "&Traceless=" + str(remove_trace) +"&gridm=" + str(grid_value) + ".npy"

print("Searching for eigenvalue collisions with grid method, for : " + grid_summary_name)

# Define the grid_search_summary dtype
grid_search_summary_dtype = np.dtype([
    ('seed', np.int32),
    ('dim', np.int32),
    ('grid_m', np.int32),
    ('unprocessed', np.int32),
    ('long_cycles', np.int32),
    ('detected_collisions', np.int32),
    ('collission_points', object)  # Placeholder for collision data
])

def refine_square_tracking(initial_matrix_type, s_0, t_0, s_1, t_1, m_fine, simplified_permutation):
    # Compute eigenvalues along the square path with finer resolution
    refined_rectangle_data = unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
        initial_matrix_type, s_0, t_0, s_1, t_0, m_fine
    )

    for (s_start, t_start, s_end, t_end) in [
        (s_1, t_0, s_1, t_1),
        (s_1, t_1, s_0, t_1),
        (s_0, t_1, s_0, t_0)
    ]:
        refined_rectangle_data = np.concatenate((
            refined_rectangle_data,
            unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
                initial_matrix_type, s_start, t_start, s_end, t_end, m_fine
            )[1:]
        ))
    
    # Extract eigenvalues
    eigenvalues = refined_rectangle_data['eigenvalues']  # (N_matrices, N_eigenvalues)
    
    # Compute minimum distances for each unique pair in simplified_permutation
    min_distances = {}
    eigenvalue_means = {}
    for cycle in simplified_permutation:
        for i, j in zip(cycle, cycle[1:] + [cycle[0]]):
            key = tuple(sorted((i, j)))  # Ensure (i, j) and (j, i) are treated the same
            min_distances[key] = {
                'side_1': np.inf, 'side_2': np.inf, 'side_3': np.inf, 'side_4': np.inf,
                'side_1_idx': None, 'side_2_idx': None, 'side_3_idx': None, 'side_4_idx': None
            }
            eigenvalue_means[key] = {'side_1': None, 'side_2': None, 'side_3': None, 'side_4': None}
    
    # Iterate over the four sides of the square
    for side_idx, (start_idx, end_idx) in enumerate([
        (0, m_fine),  # Side 1: (s_0, t_0) -> (s_1, t_0)
        (m_fine, 2 * m_fine),  # Side 2: (s_1, t_0) -> (s_1, t_1)
        (2 * m_fine, 3 * m_fine),  # Side 3: (s_1, t_1) -> (s_0, t_1)
        (3 * m_fine, 4 * m_fine)   # Side 4: (s_0, t_1) -> (s_0, t_0)
    ]):
        side_eigenvalues = eigenvalues[start_idx:end_idx]
        side_data = refined_rectangle_data[start_idx:end_idx]
        
        for cycle in simplified_permutation:
            for i, j in zip(cycle, cycle[1:] + [cycle[0]]):
                key = tuple(sorted((i, j)))
                distances = np.abs(side_eigenvalues[:, i] - side_eigenvalues[:, j])
                min_idx = np.argmin(distances)
                min_value = distances[min_idx]                
                
                if min_value < min_distances[key][f'side_{side_idx + 1}']:
                    min_distances[key][f'side_{side_idx + 1}'] = min_value
                    min_distances[key][f'side_{side_idx + 1}_idx'] = min_idx

    
    # Compute suggested solutions
    suggested_solutions = {}
    for (i, j), data in min_distances.items():
        if None not in (data['side_1_idx'], data['side_3_idx']) and None not in (data['side_2_idx'], data['side_4_idx']):
            s_collision = s_0 + (data['side_1_idx'] + data['side_3_idx']) / (2 * m_fine) * (s_1 - s_0)
            t_collision = t_0 + (data['side_2_idx'] + data['side_4_idx']) / (2 * m_fine) * (t_1 - t_0)
            eigenvalue_mean = (side_data['eigenvalues'][min_idx, i] + side_data['eigenvalues'][min_idx, j]) / 2
            suggested_solutions[(i, j)] = (s_collision, t_collision, eigenvalue_mean)
    
    return suggested_solutions

def compute_weighted_intersection(min_distances, s_0, t_0, s_1, t_1):
    """
    Compute the estimated collision point using a weighted average of the four minima.
    """
    side_positions = {
        "side_1": (np.linspace(s_0, s_1, m_fine), np.full(m_fine, t_0)),  # Bottom edge
        "side_2": (np.full(m_fine, s_1), np.linspace(t_0, t_1, m_fine)),  # Right edge
        "side_3": (np.linspace(s_1, s_0, m_fine), np.full(m_fine, t_1)),  # Top edge
        "side_4": (np.full(m_fine, s_0), np.linspace(t_1, t_0, m_fine))   # Left edge
    }

    # Extract min distances and corresponding (s, t) locations
    distances = []
    locations = []
    for side, (s_vals, t_vals) in side_positions.items():
        min_idx = min_distances[side + "_idx"]
        min_distance = min_distances[side]
        distances.append(min_distance)
        locations.append((s_vals[min_idx], t_vals[min_idx]))

    distances = np.array(distances)
    locations = np.array(locations)

    # Compute weights (inverse of distances, avoiding division by zero)
    weights = 1 / (distances + 1e-10)
    weights /= np.sum(weights)  # Normalize weights

    # Compute weighted average for collision location
    predicted_s, predicted_t = np.sum(weights[:, None] * locations, axis=0)

    return predicted_s, predicted_t

grid_search_summary_list = []
for seed in seed_list:

    # TODO: Maybe only N and features. it includes several seeds and several m per seed. 
    
    initial_matrix_type = initial_matrix_writter.generate_initial_matrix(dim, distribution, remove_trace, seed, curve)
    initial_matrix = initial_matrix_type['matrix']

    print("now processing seed " + str(seed))
    sign_matrix = np.sign(initial_matrix.real).astype(int)

    for grid_m in grid_values:
        print("now processing grid value = " + str(grid_m)) 
        dim_grid = dim * grid_m
        m_steps = 4  # Steps per subsquare
        m_fine = 16

        # Define the s, t ranges
        s_min, s_max = 0.000001, 1.0
        t_min, t_max = 0.0, 1.0

        # Compute step size for the grid
        s_step = (s_max - s_min) / dim_grid
        t_step = (t_max - t_min) / dim_grid

        # Initialize summary data
        number_of_collisions = 0
        unprocessed_squares = 0
        long_cycle_squares = 0
        eigenvalue_collissions = []
        
        for i in range(dim_grid):
            print("currently processing column " + str(i))
            print("number of collisions = " + str(number_of_collisions))
            for j in range(dim_grid):
                # Define the bounds of the current subsquare
                s_0 = s_min + i * s_step
                s_1 = s_min + (i + 1) * s_step
                t_0 = t_min + j * t_step
                t_1 = t_min + (j + 1) * t_step

                # Compute eigenvalues along the square path
                rectangle_data = unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
                    initial_matrix_type, s_0, t_0, s_1, t_0, m_steps
                )

                rectangle_data = np.concatenate((
                    rectangle_data,
                    unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
                        initial_matrix_type, s_1, t_0, s_1, t_1, m_steps
                    )[1:]
                ))

                rectangle_data = np.concatenate((
                    rectangle_data,
                    unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
                        initial_matrix_type, s_1, t_1, s_0, t_1, m_steps
                    )[1:]
                ))

                rectangle_data = np.concatenate((
                    rectangle_data,
                    unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
                        initial_matrix_type, s_0, t_1, s_0, t_0, m_steps
                    )[1:]
                ))

                # Set first item equal to

                # Order eigenvalues and refine as needed
                rectangle_data, unordered_steps = s_orderer.order_s_eigenvalues(rectangle_data)
                while_steps = 0
                while unordered_steps > 0:
                    while_steps += 1
                    rectangle_data = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, rectangle_data, curve)
                    rectangle_data, unordered_steps = s_orderer.order_s_eigenvalues(rectangle_data)
                    if while_steps == 4:
                        break
                
                if while_steps == 4:
                    unprocessed_squares += 1
                    continue            

                # Check for permutations and calculate collisions
                eigenvalues = rectangle_data['eigenvalues']  # (N_matrices, N_eigenvalues) array
                z = eigenvalues[0, :]
                w = eigenvalues[-1, :]
                permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
                cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
                simplified_permutation = find_permutation.omit_singletons(cycle_decomposition)
                cycle_length_sum = find_permutation.cycle_length_sum(simplified_permutation)
                if simplified_permutation != [] and not find_permutation.has_longer_cycles(cycle_decomposition):
                    result = refine_square_tracking(initial_matrix_type, s_0, t_0, s_1, t_1, m_fine, simplified_permutation) 
                    eigenvalue_collissions.append(result)

                if find_permutation.has_longer_cycles(cycle_decomposition):
                    long_cycle_squares +=1
                    print(f"Longer cycles found in subsquare: ({s_0}, {t_0}) to ({s_1}, {t_1})")
                    print(simplified_permutation)

                cycle_length_sum = find_permutation.cycle_length_sum(simplified_permutation)    
                number_of_collisions += cycle_length_sum

        new_entry = np.array((seed, dim, grid_m, unprocessed_squares, long_cycle_squares, number_of_collisions, eigenvalue_collissions), dtype=grid_search_summary_dtype)
        grid_search_summary_list.append(new_entry)

        # Print the total number of collisions
        print(f"Total number of collisions: {number_of_collisions}")
        print(f"Unprocessed squares: {unprocessed_squares}")

grid_search_summary_array = np.array(grid_search_summary_list, dtype=grid_search_summary_dtype)
os.makedirs(os.path.dirname(grid_summary_name), exist_ok=True)
np.save(grid_summary_name, grid_search_summary_array)

grid_search_summary_array = np.load(grid_summary_name, allow_pickle=True)
for row in grid_search_summary_array:
    print(row)

end_time = time.time()
print(f"\nTotal running time: {end_time - start_time} seconds")
print("Grid search concluded, summary saved as : " + grid_summary_name)
