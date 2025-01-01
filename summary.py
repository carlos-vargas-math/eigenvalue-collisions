import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from permutation_utils import find_permutation
from collections import defaultdict

loaded_data_summary = np.load("computed_examples/N=10&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
loaded_s_data = loaded_data_summary['summary_items']

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


