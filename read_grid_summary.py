import numpy as np
import matplotlib.pyplot as plt
from settings import settings

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
    grid_summary_name = "computed_examples/N=" + str(dim) + "&" + str(curve)  + "&Seed=" + str(seed_start)  + "&Distribution=" + distribution + "&Traceless=" + str(remove_trace) + "/gridm=" + str(grid_value) + ".npy"
else:
    grid_summary_name = "computed_examples/grid_summaries/N=" + str(dim) +"seedFrom" + str(seed_start) + "To" + str(seed_end)  + "&" + str(curve) + "&Distribution=" + distribution + "&Traceless=" + str(remove_trace) +"&gridm=" + str(grid_value) + ".npy"

grid_search_summary_array = np.load(grid_summary_name, allow_pickle=True)

last_rows = {}

for row in grid_search_summary_array:
    seed = row[0]  # Assuming the first column is the seed
    last_rows[seed] = row  # Always update to keep the last occurrence

# Print the stored last rows
for row in last_rows.values():
    row = list(row)  # Ensure row is a list
    print(" | ".join(map(str, row[:6])))  # Skip row[6]

for row in grid_search_summary_array:
    seed = row[0]  # Assuming the first column is the seed
    last_rows[seed] = row  # Always update to keep the last occurrence

collision_counts = [row[5] for row in last_rows.values()]

# Filter for even values only
min_val, max_val = min(collision_counts), max(collision_counts)
even_bins = np.arange(min_val - 1, max_val + 2, 2)  # Shift bins slightly

# Plot histogram
plt.figure(figsize=(8, 5))
hist, bins, _ = plt.hist(collision_counts, bins=even_bins, edgecolor="black", alpha=0.7)

# Center the x-ticks at the middle of each bar
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.xticks(bin_centers, labels=[int(center) for center in bin_centers])

plt.xlabel("Collision Count")
plt.ylabel("Frequency")
plt.title("Histogram of Collision Counts (complex Gaussian Ginibre to crossing)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

collision_dict = {}
total_collisions = 0
num_entries = 0

for row in last_rows.values():
    seed = row[0]  # Seed value
    collision_count = row[5]  # Collision count
    
    total_collisions += collision_count
    num_entries += 1

    if collision_count not in collision_dict:
        collision_dict[collision_count] = []
    
    collision_dict[collision_count].append(seed)

# Compute average collisions
average_collisions = total_collisions / num_entries if num_entries > 0 else 0

# Print summary
for count, seeds in sorted(collision_dict.items()):
    print(f"Collision count {count}: {len(seeds)} seeds")

print(f"\nAverage number of collisions: {average_collisions:.2f}")


# Optional: Print the dictionary
# print(collision_dict)