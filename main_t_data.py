import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import main1 as main1
import os
from settings import settings

dim = settings.dim
distribution = settings.distribution
remove_trace = settings.remove_trace
curve = settings.curve
seed = settings.seed

name = "N=" + str(dim) + "&" + str(curve) + "&Seed=" + str(seed) + "&Distribution=" + distribution + "&Traceless=" + str(remove_trace)
summary_file_name = name + ".npy"
summary_name = "computed_examples/" + name
t_steps = list(range(0, 110, 10))
# print(t_steps)

# 1001 1005 1009 1017 1020 1022 1035 1038 1043 1044 1047 1048 1070 1072 1094

# loaded_s_data = np.load("ginibre_summary_type.npy", allow_pickle=True)
# loaded_s_data = np.load("computed_examples/N=100&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
loaded_s_data = np.load(summary_file_name, allow_pickle=True)
initial_matrix_data = loaded_s_data['initial_matrix']
curve = initial_matrix_data['curve']
initial_matrix = initial_matrix_data['matrix']
s_data = loaded_s_data["summary_items"]
# Define the number of summary steps
initial_t_steps = 2000

for t_step in t_steps:
    t_data = main1.compute_t_data(t_step, initial_t_steps, initial_matrix, s_data, curve)

    # save in appropriate directory/file

    filename = "/" + str(t_step) + ".npy"
    filename_figure = summary_name + "/" + str(t_step) + ".png"

    # Ensure the directory exists
    os.makedirs(summary_name, exist_ok=True)

    # Save the file
    filename = os.path.join(summary_name, f"{t_step}.npy")
    np.save(filename, t_data)



