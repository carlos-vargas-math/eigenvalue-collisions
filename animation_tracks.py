import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from permutation_utils import find_permutation
from settings import settings, generate_directory_name

window_step = 6
zoom_in_step = -1

left_x = - 1.2
right_x = 1.2
down_y = - 1.2
up_y = 1.2
animation_filename = "animations/eigenvalue_animation"+str(window_step) +".gif"

live_steps = list(range(20*window_step, 20*(window_step+1)+1, 2))
lower_edge = 0
if window_step >= 6:
    lower_edge = window_step - 5
drawn_steps = list(range(20* lower_edge,20*window_step+ 10, 20)) 
steps = drawn_steps + live_steps
steps_2 = list(range(20,110, 20)) + list(range(102, 121, 2))

n = len(drawn_steps)  # number of steps to pre-draw
m = len(live_steps)  # number of steps to animate

assert n + m <= len(steps), "n + m must be less than or equal to number of total steps"

speed = 10

dim = settings.dim
distribution = settings.distribution
remove_trace = settings.remove_trace
curve = settings.curve
seed = settings.seed
grid_m = settings.grid_m

distribution_name = str(distribution)
summary_name = generate_directory_name()

loaded_data = [np.load(f"{summary_name}/{step}.npy", allow_pickle=True) for step in steps]
try:
    grid_search_summary_array = np.load(summary_name + "/gridm=" + str(grid_m) + ".npy", allow_pickle=True)
except FileNotFoundError:
    grid_search_summary_array = np.array([])
delta_s = 1/(settings.s_steps)
s_0=steps[-m-1]* delta_s
s_1=steps[-1]* delta_s

main_steps = range(len(steps))

data_list = [row[6] for row in grid_search_summary_array]

rows = []
for data_dict in data_list:
    for entry in data_dict:
        for (key_0, key_1), (value_0, value_1, value_2) in entry.items():
            rows.append((key_0, key_1, value_0, value_1, value_2))

# Sort the rows by s-value (third element in each tuple)
rows.sort(key=lambda x: x[2])

# Extract eigenvalues and other parameters
eigenvalues_all_steps = [data["eigenvalues"] for data in loaded_data]
s = loaded_data[0]["s"]
t = loaded_data[0]["t"]

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

# Choose a sublist of rows to include in the scatter plot
sublist_points = []
collision_count = 0
for row in rows:
    if (s_0 <= row[2] and row[2] < s_1):
        sublist_points.append((row[4].real, row[4].imag, row[3], row[2]))  # Extract (x, y, value_1)
    if (row[2] < s_1):
        collision_count +=1

# Print results
index=0
for row in sublist_points:
    print(str(index) +", " + str(row))
    index +=1


# Convert to x and y coordinates
sublist_x = [z[0] for z in sublist_points]
sublist_y = [z[1] for z in sublist_points]
sublist_value_1 = [z[2] for z in sublist_points]  # Store value_1

if (zoom_in_step in range(0,len(sublist_x))):
    window_center_x = sublist_x[zoom_in_step]
    window_center_y = sublist_y[zoom_in_step]
    window_width = 0.15
    left_x = window_center_x - window_width
    right_x = window_center_x + window_width
    down_y = window_center_y - window_width
    up_y = window_center_y + window_width
    animation_filename = "animations/eigenvalue_animation"+str(window_step) +"_"+ str(zoom_in_step) +".gif"

# Set up the plot
fig, ax = plt.subplots()
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.tick_params(colors='white')
ax.set_xlim(left_x, right_x)
ax.set_ylim(down_y, up_y)
ax.set_aspect('equal', 'box')


# Plot the additional points on the scatter plot (initially white)
scatter_special = ax.scatter(
    sublist_x, sublist_y,
    s=100,
    c="white",
    marker="x",
    alpha=0.8,
    zorder=3
)

# Plot tracks for all steps (excluding last step)
for eigenvalues, cycle_decomposition in zip(eigenvalues_all_steps[:n], all_cycle_decompositions[:n]):
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
                linewidth=0.5,
                alpha=0.7,
                color=cycle_color
            )

# Extract eigenvalues for animation (for multiple steps)
eigenx_steps = [eigenvalues_all_steps[step].real for step in main_steps]
eigeny_steps = [eigenvalues_all_steps[step].imag for step in main_steps]
colors_steps = []

# Assign colors to eigenvalues for each step
for step in main_steps:
    cycle_decomposition = all_cycle_decompositions[step]
    dim = eigenvalues_all_steps[step].shape[1]
    colors = np.zeros((dim, 4))
    for cycle in cycle_decomposition:
        cycle_size = len(cycle)
        cycle_color = size_to_color[cycle_size]
        for index in cycle:
            colors[index] = cycle_color
    colors_steps.append(colors)

# Scatter plots for multiple steps
scatter_plots = [
    ax.scatter(
        eigenx_steps[i][0, :], eigeny_steps[i][0, :],
        s=300,
        c=colors_steps[i],
        marker=".",
        edgecolor='none',
        alpha=0.7,
        zorder=2
    )
    for i in range(len(main_steps))
]

# Prepare tracks for the final step, initially empty (not drawn)
final_step_tracks = []  # Will become a list of lists

# Loop over the last m steps
for step in main_steps[-m:]:
    eigenx = eigenvalues_all_steps[step].real
    eigeny = eigenvalues_all_steps[step].imag
    decomposition = all_cycle_decompositions[step]

    step_tracks = []  # Tracks for this step

    for cycle in decomposition:
        cycle_size = len(cycle)
        cycle_color = size_to_color[cycle_size]
        for i in cycle:
            line, = ax.plot([], [], linestyle='-', linewidth=0.5, alpha=0.7, color=cycle_color)
            step_tracks.append((line, eigenx[:, i], eigeny[:, i]))

    final_step_tracks.append(step_tracks)

def animate(i):
    # Update eigenvalue positions for each step
    for j, scatter_plot in enumerate(scatter_plots):
        scatter_plot.set_offsets(np.c_[eigenx_steps[j][i, :], eigeny_steps[j][i, :]])

    # Update special scatter colors
    new_colors = []
    for value_1 in sublist_value_1:
        color = "red" if value_1 > t[i] else "white"
        new_colors.append(color)
    scatter_special.set_color(new_colors)

    # Gradually draw final step tracks up to frame i
    for step_tracks in final_step_tracks:
        for line, xdata, ydata in step_tracks:
            line.set_data(xdata[:i+1], ydata[:i+1])

    ax.set_title(f"t-step = {i}, t = {t[i]:.2f}, s in [{s_0:.2f},{s_1:.2f}], collision-count: {collision_count}", color='white')

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=range(0, eigenvalues.shape[0], speed), interval=100)
# ani = animation.FuncAnimation(fig, animate, frames=range(3490, 3491, speed), interval=100)
ani.save(animation_filename, writer="pillow", fps=10)

# Display the plot
plt.show()
