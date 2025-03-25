import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import unordered_s_increase_eigenvalue_writter as s_unordered
from computation import s_eigenvalue_orderer as s_orderer
import computation.unorderered_refinement as unorderered_refinement
from permutation_utils import find_permutation
import datatypes1
import random_matrix_model.initial_matrix_writter as initial_matrix_writter
from computation import main1
from settings import settings
import os

dim = settings.dim
distribution = settings.distribution
remove_trace = settings.remove_trace
curve = settings.curve
seed = settings.seed

summary_name = "computed_examples/N=" + str(dim) + "&" + str(curve) + "&Seed=" + str(seed) + "&" + str(distribution) + "&Traceless=" + str(remove_trace)

initial_matrix_type = initial_matrix_writter.generate_initial_matrix(dim, distribution, remove_trace, seed, curve)
initial_matrix = initial_matrix_type['matrix']

# Define the number of initial summary steps and initial rotation steps
initial_s_steps = settings.s_steps
# initial_t_steps = 1000

s_data = main1.comptute_s_data(initial_matrix, initial_s_steps, curve)

# Create a structured array instance using the Ginibre summary dtype
actual_s_steps = s_data.size
ginibre_summary_dtype = datatypes1.create_ginibre_summary_dtype(dim, actual_s_steps)
ginibre_summary = np.zeros((), dtype=ginibre_summary_dtype)

# Populate the fields
ginibre_summary['initial_matrix'] = initial_matrix_type

# Populate the Ginibre summary and save
ginibre_summary['summary_items'] = s_data

# Ensure the directory exists
os.makedirs(summary_name, exist_ok=True)

# Save the file
filename = os.path.join(summary_name, "summary.npy")
# np.save(filename, s_data)

np.save(filename, ginibre_summary)