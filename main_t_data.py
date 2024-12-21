import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import unordered_s_increase_eigenvalue_writter as s_unordered
from computation import unordered_t_increase_eigenvalue_writter as t_unordered
from computation import main1 as main1

curve = curves.Curve.CIRCLE
loaded_s_data = np.load("ginibre_summary_type.npy", allow_pickle=True)

initial_matrix_data = loaded_s_data['initial_matrix']
initial_matrix = initial_matrix_data['matrix']
s_data = loaded_s_data["summary_items"]

# Define the number of summary steps
initial_t_steps = 1000

t_data = main1.compute_t_data(601, initial_t_steps, initial_matrix, s_data, curve)

np.save('ordered_t_eigenvalues', t_data)

