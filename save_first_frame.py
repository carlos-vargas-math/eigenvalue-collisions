import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from permutation_utils import find_permutation
from collections import defaultdict

# Load the eigenvalue trajectories from the structured array

loaded_data_animation = np.load("computed_examples/N=100&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True/2.npy", allow_pickle=True)
# loaded_data_animation = np.load("ordered_t_eigenvalues.npy", allow_pickle=True)
# loaded_data_animation = np.load("well_ordered_summaries/t_eigenvaluesS100.npy", allow_pickle=True)

# Extract data from the loaded array
s = loaded_data_animation["s"]
t = loaded_data_animation["t"]
eigenvalues = loaded_data_animation['eigenvalues']  # (N_matrices, N_eigenvalues) array of complex eigenvalues

# Initial eigenvalues
z = eigenvalues[0, :]
w = eigenvalues[-1, :]
permuted_w, permutation_indices = find_permutation.find_best_permutation(z, w)
cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)

# Separate the real and imaginary parts
eigenx = eigenvalues.real  # Real part of the eigenvalues
eigeny = eigenvalues.imag  # Imaginary part of the eigenvalues

# Marker size and initial settings
marker_size = 300
initial_axes = 1.2

# Get unique cycle sizes and their indices
cycle_sizes = [len(cycle) for cycle in cycle_decomposition]  # Sizes of each cycle
unique_sizes = sorted(set(cycle_sizes))  # Sorted unique cycle sizes
num_unique_sizes = len(unique_sizes)  # Number of unique sizes

# Check if all cycles are singletons by verifying the number of unique sizes and the size of the first cycle
all_singletons = num_unique_sizes == 1 and len(cycle_decomposition[0]) == 1

# Define colors
singleton_color = (1.0, 0.6, 0.0, 1.0) if all_singletons else (1.0, 1.0, 0.0, 1.0)  # Warmer orange or yellow
cmap = plt.get_cmap("plasma")

# Map each unique size to an evenly spaced color
size_to_color = {
    size: singleton_color if size == 1 else cmap(0.25 + 0.75 * (1 - i / (num_unique_sizes - 1)))
    for i, size in enumerate(unique_sizes)
}

# Assign colors to eigenvalues based on their cycle's size
dim = eigenvalues.shape[1]
colors = np.zeros((dim, 4))  # RGBA array for each eigenvalue
for cycle in cycle_decomposition:
    cycle_size = len(cycle)
    cycle_color = size_to_color[cycle_size]  # Get the color for this cycle size
    for index in cycle:
        colors[index] = cycle_color

# Method to save the first frame as a PNG image
def save_first_frame(filename="first_frame.png", displayOnlySmallCycles=None):
    # Clear the current figure and axes
    fig, ax = plt.subplots()
    
    # Set the background to dark and adjust other colors
    ax.set_facecolor("black")  # Dark background for the plot
    fig.patch.set_facecolor("black")  # Dark background for the entire figure
    ax.tick_params(colors='white')  # White ticks
    # Hide axis labels and ticks
    ax.axis('off')

    # Set limits and aspect ratio
    ax.set_xlim(-initial_axes, initial_axes)
    ax.set_ylim(-initial_axes, initial_axes)
    ax.set_aspect('equal', 'box')

    # Scatter plot with the updated colors
    scatter_plot = ax.scatter(
        eigenx[0, :], 
        eigeny[0, :], 
        s=marker_size, 
        c=colors,  # Directly use RGBA values
        marker=".", 
        edgecolor='none',
        zorder=3
    )

    # Annotate eigenvalues with the next index in their cycle
    for cycle in cycle_decomposition:
        # Iterate through the cycle in order
        for i in range(len(cycle)):
            current_index = cycle[i]
            ax.text(
                eigenx[0, current_index], 
                eigeny[0, current_index], 
                str(cycle[i]),  # Label with the next index in the cycle
                color='black', 
                fontsize=6, 
                ha='center', 
                va='center',
                zorder=3 
            )

    # Plot trajectories with cycle-specific colors or arrows for large cycles
    step_interval = 1
    # Plot trajectories with cycle-specific colors or line segments for large cycles
    for cycle in cycle_decomposition:
        cycle_size = len(cycle)
        cycle_color = size_to_color[cycle_size]

        if displayOnlySmallCycles is not None and cycle_size > displayOnlySmallCycles:
            # Replace the trajectory with a line segment for larger cycles
            for i in cycle:
                start_points = eigenvalues[0, i]
                end_points = eigenvalues[-1, i]

                # Draw a line segment between the average start and end points
                print(f"Start: {start_points}, End: {end_points}")
                ax.plot(
                    [start_points.real, end_points.real],
                    [start_points.imag, end_points.imag],
                    linestyle='-', linewidth=1.0, alpha=1.0, color=cycle_color, zorder=1
                )
        else:
            # Plot detailed trajectory for small cycles
            for i in cycle:
                ax.plot(
                    eigenx[::step_interval, i],
                    eigeny[::step_interval, i],
                    linestyle='-', linewidth=0.5, alpha=0.7, color=cycle_color, zorder=1
                )

    # Set a title
    ax.set_title(f"Time step = 0, t = {t[0]:.4f}, s = {s[0]:.4f}", color='white')

    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"First frame saved as {filename}")
    plt.close(fig)

# Example usage
name = "N=" + str(dim)+"step"+str(10*dim) +".png"
save_first_frame("crossing", displayOnlySmallCycles=100)
