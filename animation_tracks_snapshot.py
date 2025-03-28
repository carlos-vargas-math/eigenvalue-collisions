import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from permutation_utils import find_permutation

# Load data for the required steps circle, N=10, seed 1000
# steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 1  # collision slice indices
# steps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# n, m = 1, 10  # collision slice indices
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 300]
# n, m = 10, 17  # collision slice indices
# steps = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
# n, m = 17, 43  # collision slice indices
steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
n, m = 43, 71  # collision slice indices
# steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
# n, m = 71, 85  # collision slice indices
# steps = [600, 620, 640, 660, 680, 700]
# n, m = 85, 90  # collision slice indices
loaded_data = [np.load(f"computed_examples/N=10&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True/{step}.npy", allow_pickle=True) for step in steps]
grid_search_summary_array = np.load("N=10seedFrom1000To1000&Curve.CIRCLE&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)


# Load data for the required steps ciruit, N=10, seed 1000
# steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 3  # collision slice indices
# steps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# n, m = 3, 7  # collision slice indices
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 300]
# n, m = 7, 12  # collision slice indices
# steps = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
# n, m = 12, 25  # collision slice indices
# steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
# n, m = 25, 53  # collision slice indices
# steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
# n, m = 53, 87  # collision slice indices
# steps = [600, 620, 640, 660, 680, 700]
# n, m = 87, 100  # collision slice indices
# loaded_data = [ np.load(f"computed_examples/N=100&Curve.CIRCUIT&Seed=1000&Distribution=complexGaussian&Traceless=True/{step}.npy", allow_pickle=True) for step in steps]
# grid_search_summary_array = np.load("N=10seedFrom1000To1000&Curve.CIRCUIT&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)

# Load data for the required steps crossing, N=11, seed 1001, Traceless=False
# steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 0  # collision slice indices
# steps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] # OK!!
# n, m = 0, 0   # collision slice indices
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 300]
# n, m = 3, 3  # collision slice indices
# steps = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
# n, m = 19, 19  # collision slice indices
# steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
# n, m = 46, 46  # collision slice indices
# steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
# n, m = 81, 81  # collision slice indices
# steps = [600, 620, 640, 660, 680, 700]
# n, m = 100, 100  # collision slice indices
# steps = [700, 720, 740, 760, 780, 800]
# n, m = 100, 100  # collision slice indices
# steps = [800, 820, 840, 860, 880, 900]
# n, m = 100, 100  # collision slice indices
# loaded_data = [np.load(f"computed_examples/N=11&Curve.CROSSING&Seed=1001&Distribution=opposingSectors&Traceless=False/{step}.npy", allow_pickle=True) for step in steps]
# grid_search_summary_array = np.load("N=11seedFrom1001To1001&Curve.CROSSING&Distribution=opposingSectors&Traceless=False10.npy", allow_pickle=True)
# grid_search_summary_array = []


# Load data for the required steps bernoulli, circle, N=10, seed 1001, Traceless = True
# steps = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 9  # collision slice indices  # first three collissions count to but one is ill-located
# steps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# n, m = 9, 16  # collision slice indices
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 300]
# n, m = 16, 26  # collision slice indices
# steps = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
# n, m = 26, 36  # collision slice indices
# steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
# n, m = 36, 57  # collision slice indices
# steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
# n, m = 57, 82  # collision slice indices
# steps = [600, 620, 640, 660, 680, 700]
# n, m = 82, 88  # collision slice indices
# loaded_data = [np.load(f"computed_examples/N=10&Curve.CIRCLE&Seed=1001&Distribution=bernoulli&Traceless=True/{step}.npy", allow_pickle=True) for step in steps]
# grid_search_summary_array = np.load("N=10seedFrom1001To1001&Curve.CIRCLE&Distribution=bernoulli&Traceless=True20.npy", allow_pickle=True)


# Load data for the required steps bernoulli, circle, N=10, seed 1070, Traceless = True
# steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 4  # collision slice indices  # first three collissions count to but one is ill-located
# steps = [100, 110, 120, 130, 140, 150, 159, 160, 170, 180, 190, 200]
# n, m = 4, 11  # collision slice indices
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 300]
# n, m = 11, 21  # collision slice indices
# steps = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
# n, m = 26, 36  # collision slice indices
# steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
# n, m = 36, 57  # collision slice indices
# steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
# n, m = 57, 82  # collision slice indices
# steps = [600, 620, 640, 660, 680, 700]
# n, m = 82, 88  # collision slice indices
# loaded_data = [np.load(f"computed_examples/N=10&Curve.CIRCLE&Seed=1070&Distribution=bernoulli&Traceless=True/{step}.npy", allow_pickle=True) for step in steps]
# grid_search_summary_array = np.load("N=10seedFrom1070To1070&Curve.CIRCLE&Distribution=bernoulli&Traceless=True20.npy", allow_pickle=True)


# Load data for the required steps crossing, N=10, seed 1001, Traceless=False
# steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 0  # collision slice indices
# steps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] # OK!!
# n, m = 0, 0   # collision slice indices
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 300]
# n, m = 3, 3  # collision slice indices
# steps = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
# n, m = 19, 19  # collision slice indices
# steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
# n, m = 46, 46  # collision slice indices
# steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
# n, m = 81, 81  # collision slice indices
# steps = [600, 620, 640, 660, 680, 700]
# n, m = 100, 100  # collision slice indices
# steps = [700, 720, 740, 760, 780, 800]
# n, m = 100, 100  # collision slice indices
# steps = [800, 820, 840, 860, 880, 900]
# n, m = 100, 100  # collision slice indices
# loaded_data = [np.load(f"computed_examples/N=10&Curve.CROSSING&Seed=1001&Distribution=opposingSectors&Traceless=False/{step}.npy", allow_pickle=True) for step in steps]
# grid_search_summary_array = np.load("N=10seedFrom1001To1001&Curve.CROSSING&Distribution=opposingSectors&Traceless=False10.npy", allow_pickle=True)



# Load data for the required steps ciruit + meander, N=10, seed 1000
# steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 0  # collision slice indices
# steps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# n, m = 0, 0  # collision slice indices
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 300]
# n, m = 0, 9  # collision slice indices
# steps = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
# n, m = 9, 30  # collision slice indices
# steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
# n, m = 30, 58  # collision slice indices
# steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
# n, m = 58, 89  # collision slice indices
# steps = [600, 620, 640, 660, 680, 700]
# n, m = 89, 99  # collision slice indices
# steps = [700, 720, 740, 760, 780, 800]
# n, m = 99, 100  # collision slice indices
# loaded_data = [ np.load(f"computed_examples/N=10&Curve.CIRCUIT&Seed=1000&Distribution=ginibreMeander&Traceless=False/{step}.npy", allow_pickle=True) for step in steps]
# grid_search_summary_array = np.load("N=10seedFrom1000To1000&Curve.CIRCUIT&Distribution=ginibreMeander&Traceless=False.npy", allow_pickle=True)



# Load data for the required steps circle, N=100, seed 1000
# steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 1  # collision slice indices
# steps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# n, m = 1, 10  # collision slice indices
# steps = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 300]
# n, m = 10, 17  # collision slice indices
# steps = [300, 310, 320, 330, 340, 350]
# n, m = 17, 43  # collision slice indices
# steps = [350, 360, 370, 380, 390, 400]
# n, m = 17, 43  # collision slice indices
# steps = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
# n, m = 43, 71  # collision slice indices
# steps = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
# n, m = 71, 85  # collision slice indices
# steps = [600, 620, 640, 660, 680, 700]
# n, m = 85, 90  # collision slice indices
# loaded_data = [ np.load(f"computed_examples/N=100&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True/{step}.npy", allow_pickle=True) for step in steps]
# grid_search_summary_array = []

# Load data for the required steps crossing, N=11, seed 1001, Traceless=False
# steps = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n, m = 0, 0  # collision slice indices
# loaded_data = [np.load(f"computed_examples/N=10&Curve.CIRCLE&Seed=1017&Distribution=bernoulli&Traceless=True/{step}.npy", allow_pickle=True) for step in steps]
# grid_search_summary_array = []

# 1001 1005 1009 1017 1020 1022 1035 1038 1043 1044 1047 1048 1070 1072 1094

# non main steps scatter points are not displayed. 
speed = 5
main_steps = range(len(steps))

# Extract the dictionaries from the data
data_list = [row[6] for row in grid_search_summary_array]

rows = []
for data_dict in data_list:  # data_list contains lists of dictionaries
    for entry in data_dict:  # Iterate over the list of dictionaries
        for (key_0, key_1), (value_0, value_1, value_2) in entry.items():
            rows.append((key_0, key_1, value_0, value_1, value_2))

# Sort the rows by value_0 (third element in each tuple)
rows.sort(key=lambda x: x[2])

# Print results
for row in rows:
    print(row)


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

# Set up the plot
fig, ax = plt.subplots()
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.tick_params(colors='white')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal', 'box')

# Choose a sublist of rows to include in the scatter plot
sublist_points = [(row[4].real, row[4].imag, row[3]) for row in rows[n:m]]  # Extract (x, y, value_1)

# Convert to x and y coordinates
sublist_x = [z[0] for z in sublist_points]
sublist_y = [z[1] for z in sublist_points]
sublist_value_1 = [z[2] for z in sublist_points]  # Store value_1

# Plot the additional points on the scatter plot (initially white)
scatter_special = ax.scatter(
    sublist_x, sublist_y,
    s=20,
    c="white",
    marker="x",
    alpha=0.8,
    zorder=3
)

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
                linewidth=0.2,
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
        s=50,
        c=colors_steps[i],
        marker=".",
        edgecolor='none',
        alpha=0.7,
        zorder=2
    )
    for i in range(len(main_steps))
]

fig.savefig("initial_frame.pdf", format="pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")

def animate(i):
    for j, scatter_plot in enumerate(scatter_plots):
        scatter_plot.set_offsets(np.c_[eigenx_steps[j][i, :], eigeny_steps[j][i, :]])

    new_colors = []
    for value_1 in sublist_value_1:
        color = "red" if value_1 > t[i] else "white"
        new_colors.append(color)

    scatter_special.set_color(new_colors)  # Update colors dynamically

    ax.set_title(f"Time step = {i}, t = {t[i]:.4f}, s = {s[i]:.4f}", color='white')


# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=range(0, eigenvalues.shape[0], speed), interval=100)

# Display the plot
plt.show()
