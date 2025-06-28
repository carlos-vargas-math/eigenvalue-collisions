import random_matrix_model.points_on_curve as curves
import numpy as np
from computation import main1 as main1
import os
from settings import settings, generate_directory_name

dim = settings.dim
distribution = settings.distribution
remove_trace = settings.remove_trace
curve = settings.curve
seed = settings.seed

summary_name = generate_directory_name()
summary_file_name = summary_name + "/summary.npy"

t_steps = list(range(160,481, 2))
# t_steps = list(range(1000,2010, 20))
# t_steps = list(range(111, 120, 1))
# t_steps = [1000, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
# t_steps = [1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179]


loaded_s_data = np.load(summary_file_name, allow_pickle=True)
initial_matrix_data = loaded_s_data['initial_matrix']
curve = initial_matrix_data['curve']
initial_matrix = initial_matrix_data['matrix']
s_data = loaded_s_data["summary_items"]
# Define the number of summary steps
initial_t_steps = settings.t_steps

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



