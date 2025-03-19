import numpy as np
import matplotlib.pyplot as plt
# grid_search_summary_array = np.load("N=10seedFrom1000To1099&Curve.CIRCLE&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=10seedFrom1000To1049&Curve.CIRCUIT&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=20seedFrom1000To1005&Curve.CIRCLE&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=20seedFrom1000To1001&Curve.CIRCUIT&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=10seedFrom1051To1100&Curve.CIRCUIT&Distribution=ginibreMeander&Traceless=False.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=10seedFrom2000To2100&Curve.CIRCUIT&Distribution=ginibreMeander&Traceless=False5.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=10seedFrom2000To2100&Curve.CIRCUIT&Distribution=complexGaussian&Traceless=False.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=10seedFrom2000To2100&Curve.CIRCUIT&Distribution=complexGaussian&Traceless=False5.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=10seedFrom2000To2100&Curve.CIRCLE&Distribution=complexGaussian&Traceless=False5.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=10seedFrom1000To1100&Curve.CIRCLE&Distribution=bernoulli&Traceless=True5.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=11seedFrom1000To1010&Curve.CROSSING&Distribution=opposingSectors&Traceless=False6.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=11seedFrom1000To1010&Curve.CROSSING&Distribution=opposingSectors&Traceless=True6.npy", allow_pickle=True)
# grid_search_summary_array = np.load("N=11seedFrom1000To1100&Curve.CROSSING&Distribution=complexGaussian&Traceless=False5.npy", allow_pickle=True)
grid_search_summary_array = np.load("N=11seedFrom1000To1100&Curve.CROSSING&Distribution=opposingSectors&Traceless=False5.npy", allow_pickle=True)

# grid_search_summary_array = np.load('grid_search_summary/grid_search_summary.npy', allow_pickle=True)
# grid_search_summary_array = np.load('grid_search_summary/grid_search_summaryN5Circuit.npy', allow_pickle=True)

# for row in grid_search_summary_array:  
#     print(str(row[0]) + " | " + str(row[1]) + " | " + str(row[2]) + " | "  + str(row[3]) + " | "  + str(row[4]) + " | " + str(row[5]))

for row in grid_search_summary_array:
    if (row[5] != 90):  
        print(str(row[0]) + " | " + str(row[1]) + " | " + str(row[2]) + " | "  + str(row[3]) + " | "  + str(row[4]) + " | " + str(row[5]))


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