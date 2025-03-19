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

# 1001 1005 1009 1017 1020 1022 1035 1038 1043 1044 1047 1048 1070 1072 1094

# define parameter values
dim = 10
# distribution = 'complexGaussian'
distribution = 'bernoulli'
remove_trace = True
curve = curves.Curve.CIRCLE
seed = 1095

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