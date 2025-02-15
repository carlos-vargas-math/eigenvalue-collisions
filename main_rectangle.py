import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import unordered_s_increase_eigenvalue_writter as s_unordered
from computation import s_eigenvalue_orderer as s_orderer
import computation.unorderered_refinement as unorderered_refinement
from permutation_utils import find_permutation
import datatypes1
import random_matrix_model.initial_matrix_writter as initial_matrix_writter
from computation import main1
from computation_rect import unordered_linear_segment_eigenvalue_writter
import numpy as np

# set initial matrix and settings

# define parameter values
dim = 40
distribution = 'complexGaussian'
remove_trace = True
curve = curves.Curve.CIRCLE
seed = 1000

compute_summary = True
summary_name = "N=" + str(dim) + "&" + str(curve) + "&Seed=" + str(seed) + "&Distribution=" + distribution + "&Traceless=" + str(remove_trace)

initial_matrix_type = initial_matrix_writter.generate_initial_matrix(dim, distribution, remove_trace, seed, curve)
initial_matrix = initial_matrix_type['matrix']

number_of_colissions = 0

# will calculate along square path p_0 = (s_0, t_0) --> p_1 = (s_1, t_0) --> p_2(s_1, t_1) --> p_3(s_0, t_1) --> p_0.
# m = int  ceiling (10000 * (s_1 - s_0)) 
s_0 = 0.1
s_1 = 0.2
t_0 = 0.5
t_1 = 0.6
m=1000

rectangle_data_1 = unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
    initial_matrix_type, s_0, t_0, s_1, t_0, m
)

rectangle_data_1 = np.concatenate((
    rectangle_data_1,
    unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
        initial_matrix_type, s_1, t_0, s_1, t_1, m
    ) [1:]
))

rectangle_data_1 = np.concatenate((
    rectangle_data_1,
    unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
        initial_matrix_type, s_1, t_1, s_0, t_1, m
    ) [1:]
))

rectangle_data_1 = np.concatenate((
    rectangle_data_1,
    unordered_linear_segment_eigenvalue_writter.get_eigenvalues_along_segment(
        initial_matrix_type, s_0, t_1, s_0, t_0, m
    ) [1:]
))

rectangle_data_1, unordered_steps = s_orderer.order_s_eigenvalues(rectangle_data_1)
while unordered_steps > 0:
    rectangle_data_1 = unorderered_refinement.insert_unordered_refinement_points(initial_matrix, rectangle_data_1, curve)
    rectangle_data_1, unordered_steps = s_orderer.order_s_eigenvalues(rectangle_data_1)


eigenvalues = rectangle_data_1['eigenvalues']  # This is a (N_matrices, N_eigenvalues) array of complex eigenvalues
z = eigenvalues[0, :]
w = eigenvalues[-1, :]
permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
simplified_permuation = find_permutation.omit_singletons(difference_permutation)

print(cycle_decomposition)
if (find_permutation.has_longer_cycles(cycle_decomposition)):
    # Todo: iterate method along in 4 subsquares
    print("todo")
else:
    cycle_length_sum = find_permutation.cycle_length_sum(simplified_permuation)
    number_of_colissions += cycle_length_sum
