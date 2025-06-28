import numpy as np
import matplotlib.pyplot as plt
from settings import settings, generate_directory_name

# For statistics of multiple seed values, set load_parameters_from_settings = False
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

grid_search_summary_array = np.load(grid_summary_name, allow_pickle=True)


# Initialize empty list to collect the values
values = []

# Extract the first element of each value in row[6]
for row in grid_search_summary_array:
    print(str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4]) + "," + str(row[5]))
    for item in row[6]:  # item is a dict with one key-value pair
        for val in item.values():
            values.append(val[0])  # val is a tuple; val[0] is the number we're interested in

# Create histogram
plt.hist(values, bins=20, range=(0, 1), edgecolor='black')
plt.title("Histogram of 1560 collisions by s-value (N = 40)")
plt.xlabel("s")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
