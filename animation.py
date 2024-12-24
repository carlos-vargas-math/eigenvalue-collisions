import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from permutation_utils import find_permutation
from collections import defaultdict

# Load the eigenvalue trajectories from the structured array
loaded_data_animation = np.load("ordered_t_eigenvalues.npy", allow_pickle=True)
# loaded_data_animation = np.load("well_ordered_summaries/t_eigenvaluesS100.npy", allow_pickle=True)

# loaded_data_summary = np.load("ordered_s_eigenvalues.npy", allow_pickle=True)
# loaded_data = np.load("summary/ordered_s_eigenvalues.npy", allow_pickle=True)
# loaded_data = np.load("summary/complexGaussian1.npy", allow_pickle=True)
loaded_data_summary = np.load("computed_examples/N=10&Curve.CIRCUIT&Seed=1001&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
loaded_s_data = loaded_data_summary['summary_items']
# loaded_s_data = np.load("ordered_s_eigenvalues.npy")
# loaded_data = np.load("well_ordered_summaries/complexGaussianN5.npy", allow_pickle=True)
collisions = 0
cycle_counts = defaultdict(int)

for step in range(10, 999): 
    z = loaded_s_data['eigenvalues'][step]
    permutation_indices = loaded_s_data['associated_permutation'][step]
    permutation_indices_next = loaded_s_data['associated_permutation'][step + 1]
    cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
    difference_permutation = find_permutation.find_resultant_permutation(permutation_indices, permutation_indices_next)
    print(cycle_decomposition)
    print(find_permutation.omit_singletons(difference_permutation))
    print("")

 # Count the cycles in difference_permutation
    for cycle in find_permutation.omit_singletons(difference_permutation):
        # Convert cycle to tuple for dictionary key
        print('cycle_decomposition ' + str(cycle_decomposition)  + ' at step ' + str(step))
        print(cycle)
        cycle_key = tuple(cycle)
        cycle_counts[cycle_key] += 1

    # increase counters for cycles in difference_permutation 
    collisions += find_permutation.cycle_length_sum(find_permutation.omit_singletons(difference_permutation))
print()    

print("collisions")
print(collisions)


# Sort the cycles by frequency (value) in descending order
sorted_cycle_counts = sorted(cycle_counts.items(), key=lambda item: item[1], reverse=True)

# Print the sorted cycles with their counts
print("Sorted Cycle Counts (by frequency):")
for cycle, count in sorted_cycle_counts:
    print(f"{cycle}: {count}")

print(loaded_s_data['associated_permutation'][10])
print(loaded_s_data['associated_permutation'][11])

s = loaded_data_animation["s"]
t = loaded_data_animation["t"]
# Extract eigenvalues from the 'eigenvalues' field in the structured array
eigenvalues = loaded_data_animation['eigenvalues']  # This is a (N_matrices, N_eigenvalues) array of complex eigenvalues

# color_by_abs = True
color_by_abs = False

z = eigenvalues[0, :]
w = eigenvalues[-1, :]
permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
print(permutation_indices)
cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
print(cycle_decomposition)

# Separate the real and imaginary parts
eigenx = eigenvalues.real  # Real part of the eigenvalues
eigeny = eigenvalues.imag  # Imaginary part of the eigenvalues

# Number of time steps based on the number of matrices in the process
time_steps = eigenvalues.shape[0]

# Number of eigenvalues (dimension of each matrix)
dim = eigenvalues.shape[1]

# Marker size and color initialization using a colormap
marker_size = 300
num_cycles = len(cycle_decomposition)


# Set up the plot
initial_axes = 1.2
fig1, ax1 = plt.subplots()

# Set the background to dark and adjust other colors
ax1.set_facecolor("black")  # Dark background for the plot
fig1.patch.set_facecolor("black")  # Dark background for the entire figure
ax1.tick_params(colors='white')  # White ticks

# Set limits and aspect ratio
ax1.set_xlim(-initial_axes, initial_axes)
ax1.set_ylim(-initial_axes, initial_axes)
ax1.set_aspect('equal', 'box')

# Get unique cycle sizes and their indices
cycle_sizes = [len(cycle) for cycle in cycle_decomposition]  # Sizes of each cycle
unique_sizes = sorted(set(cycle_sizes))  # Sorted unique cycle sizes
num_unique_sizes = len(unique_sizes)  # Number of unique sizes

# Map each unique size to an evenly spaced color
cmap = plt.get_cmap("plasma")
size_to_color = {size: cmap(1 - i / (num_unique_sizes - 1)) for i, size in enumerate(unique_sizes)}

# Assign colors to eigenvalues based on their cycle's size
colors = np.zeros((dim, 4))  # RGBA array for each eigenvalue
for cycle in cycle_decomposition:
    cycle_size = len(cycle)
    cycle_color = size_to_color[cycle_size]  # Get the color for this cycle size
    for index in cycle:
        colors[index] = cycle_color

# Scatter plot with the updated colors
scatter_plot = ax1.scatter(
    eigenx[0, :], 
    eigeny[0, :], 
    s=marker_size, 
    c=colors,  # Directly use RGBA values
    marker=".", 
    edgecolor='none'
)

# Plot trajectories with cycle-specific colors
# step_interval = 1
# for cycle in cycle_decomposition:
#     cycle_size = len(cycle)
#     cycle_color = size_to_color[cycle_size]
#     for i in cycle:
#         ax1.plot(
#             eigenx[::step_interval, i], 
#             eigeny[::step_interval, i], 
#             linestyle='-', linewidth=0.5, alpha=0.7, color=cycle_color
#         )

# Define the animate function as before

# Define k: skip every k-1 frames
k = 10

# Update function for animation
def animate(i):
    """Perform animation step."""
    # Update scatter plot data for the i-th time step
    scatter_plot.set_offsets(np.c_[eigenx[i, :], eigeny[i, :]])
    
    # Update title or text with a bright color
    ax1.set_title(f"Time step = {i}, t = {t[i]:.4f}, s = {s[i]:.4f}", color='white')


# Create the animation
ani1 = animation.FuncAnimation(fig1, animate, frames=range(0, time_steps, k), interval=100)

# Display the plot
plt.show()
