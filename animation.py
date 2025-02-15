import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from permutation_utils import find_permutation

# loaded_data_animation = np.load("computed_examples/LargeN/s=151&N=400&Curve.CIRCLE&Seed=999&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
loaded_data_animation = np.load("ordered_t_eigenvalues.npy", allow_pickle=True)
# loaded_data_animation = np.load("computed_examples/N=10&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True/300.npy", allow_pickle=True)

s = loaded_data_animation["s"]
t = loaded_data_animation["t"]

eigenvalues = loaded_data_animation['eigenvalues']  
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
size_to_color = {
    size: cmap(0.25) if len(unique_sizes) == 1 else cmap(0.25 + 0.75 * (1 - i / (num_unique_sizes - 1)))
    for i, size in enumerate(unique_sizes)
}

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
step_interval = 1
for cycle in cycle_decomposition:
    cycle_size = len(cycle)
    cycle_color = size_to_color[cycle_size]
    for i in cycle:
        ax1.plot(
            eigenx[::step_interval, i], 
            eigeny[::step_interval, i], 
            linestyle='-', linewidth=0.5, alpha=0.7, color=cycle_color
        )

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
