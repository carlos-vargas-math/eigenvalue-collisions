import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import main1 as main1
from random_matrix_model import points_on_curve

# loaded_s_data = np.load("ginibre_summary_type.npy", allow_pickle=True)
loaded_s_data = np.load("N=10&Curve.CIRCLE&Seed=997&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
initial_matrix_data = loaded_s_data['initial_matrix']
curve = initial_matrix_data['curve']
# curve = points_on_curve.Curve.CIRCLE
initial_matrix = initial_matrix_data['matrix']
s_data = loaded_s_data["summary_items"]

# Define the number of summary steps
initial_t_steps = 2000

t_data = main1.compute_t_data(648, initial_t_steps, initial_matrix, s_data, curve)

np.save('ordered_t_eigenvalues', t_data)

