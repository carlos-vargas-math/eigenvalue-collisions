import numpy as np
import matplotlib.pyplot as plt
from permutation_utils import find_permutation

# Load data for the required steps
# steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# steps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
# steps = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
# steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]

# steps = [50, 60, 70, 80, 90, 100]
# steps = [100, 110, 120, 130, 140, 150]
# steps = [150, 160, 170, 180, 190, 200]
# steps = [200, 210, 220, 230, 240, 250]
# steps = [250, 260, 270, 280, 290, 300]

main_step = 3
loaded_data = [
    np.load(f"computed_examples/N=10&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True/{step}.npy", allow_pickle=True)
    for step in steps
]

# Extract eigenvalues and other parameters
eigenvalues_all_steps = [data["eigenvalues"] for data in loaded_data]
s = loaded_data[main_step]["s"]
t = loaded_data[main_step]["t"]

# Gather cycle decompositions across all steps
all_cycle_sizes = set()
all_cycle_decompositions = []
for eigenvalues in eigenvalues_all_steps:
    z = eigenvalues[0, :]
    w = eigenvalues[-1, :]
    _, permutation_indices = find_permutation.find_best_permutation(z, w)
    cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
    all_cycle_decompositions.append(cycle_decomposition)
    all_cycle_sizes.update(len(cycle) for cycle in cycle_decomposition)

# Prepare for coloring based on cycle sizes
unique_sizes = sorted(all_cycle_sizes)
num_unique_sizes = len(unique_sizes)
cmap = plt.get_cmap("plasma")
size_to_color = {
    size: cmap(0.25) if num_unique_sizes == 1 else cmap(0.25 + 0.75 * (1 - i / (num_unique_sizes - 1)))
    for i, size in enumerate(unique_sizes)
}

# Set up the plot with reduced margins
fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figsize to increase the figure size
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Reduce padding around the plot

ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.tick_params(colors='white')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal', 'box')

# Plot tracks for all steps
for eigenvalues, cycle_decomposition in zip(eigenvalues_all_steps, all_cycle_decompositions):
    eigenx = eigenvalues.real
    eigeny = eigenvalues.imag
    for cycle in cycle_decomposition:
        cycle_size = len(cycle)
        cycle_color = size_to_color[cycle_size]
        for i in cycle:
            ax.plot(
                eigenx[:, i],
                eigeny[:, i],
                linestyle='-',
                linewidth=0.8,
                alpha=0.7,
                color=cycle_color
            )

# Extract eigenvalues for step 95
eigenvalues = eigenvalues_all_steps[main_step]  # Step 95
eigenx = eigenvalues.real
eigeny = eigenvalues.imag
cycle_decomposition = all_cycle_decompositions[main_step]
dim = eigenvalues.shape[1]

# Assign colors to eigenvalues for scatter plot
colors = np.zeros((dim, 4))
for cycle in cycle_decomposition:
    cycle_size = len(cycle)
    cycle_color = size_to_color[cycle_size]
    for index in cycle:
        colors[index] = cycle_color

# Scatter plot for the selected frame
scatter_plot = ax.scatter(
    eigenx[0, :], eigeny[0, :],
    s=15,
    c=colors,
    marker=".",
    edgecolor='none'
)

# Add arrows to indicate direction of movement with fixed size
arrow_offsets_x = eigenx[10, :] - eigenx[0, :]
arrow_offsets_y = eigeny[10, :] - eigeny[0, :]

# Normalize the direction vectors to a fixed length
arrow_magnitude = np.sqrt(arrow_offsets_x**2 + arrow_offsets_y**2)
fixed_arrow_length = 0.006  # Adjust this value for arrow size
normalized_arrow_offsets_x = (arrow_offsets_x / arrow_magnitude) * fixed_arrow_length
normalized_arrow_offsets_y = (arrow_offsets_y / arrow_magnitude) * fixed_arrow_length

# Plot the arrows
arrows = ax.quiver(
    eigenx[0, :], eigeny[0, :],  # Starting points
    normalized_arrow_offsets_x, normalized_arrow_offsets_y,  # Normalized direction vectors
    angles='xy', scale_units='xy', scale=1, color="black", alpha=0.7, width=0.002, zorder=3
)

# Add title for the frame
ax.set_title(f"Time step = 0, t = {t[0]:.4f}, s = {s[0]:.4f}", color='white')

# Save as vector image
plt.savefig("frame95_with_arrows.svg", format="svg")  # Save as SVG
plt.savefig("frame95_with_arrows.pdf", format="pdf")  # Or save as PDF

plt.show()
