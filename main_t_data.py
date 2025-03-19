import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import main1 as main1
import os
summary_file_name = "N=10&Curve.CIRCLE&Seed=1095&Distribution=bernoulli&Traceless=True.npy"
summary_name = "computed_examples/N=10&Curve.CIRCLE&Seed=1095&Distribution=bernoulli&Traceless=True"
# t_steps = [200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250]
# t_steps = [250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]
t_steps = [10,20,30,40,50]
# t_steps = list(range(10, 110, 10))
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



