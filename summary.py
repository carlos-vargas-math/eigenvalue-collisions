import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from permutation_utils import find_permutation
from collections import defaultdict

loaded_data_summary = np.load("N=10&Curve.CIRCLE&Seed=1000&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
# loaded_s_data = np.load("refined_ordered_s_eigenvalues.npy", allow_pickle=True)
loaded_s_data = loaded_data_summary['summary_items']
actual_s_steps = loaded_s_data.size

collisions = 0
first_collision_step = None
last_collision_step = None
cycle_counts = defaultdict(int)
max_cycle_length_sum = 0  # Variable to store the maximum cycle_length_sum
max_cycle_length_sum_step = None  # Variable to store the step with the maximum cycle_length_sum
store_difference_permutation = None
total_steps_with_long_cycles = 0

for step in range(1, actual_s_steps - 1): 
    z = loaded_s_data['eigenvalues'][step]
    permutation_indices = loaded_s_data['associated_permutation'][step]
    permutation_indices_next = loaded_s_data['associated_permutation'][step + 1]
    cycle_decomposition = find_permutation.cycle_decomposition(permutation_indices)
    difference_permutation = find_permutation.find_resultant_permutation(permutation_indices, permutation_indices_next)
    print(cycle_decomposition)
    print(find_permutation.omit_singletons(difference_permutation))
    print("")
    if any(len(cycle) > 1 for cycle in difference_permutation):
        if first_collision_step is None:
            first_collision_step = step
        last_collision_step = step
    if any(len(cycle) > 2 for cycle in difference_permutation):
        total_steps_with_long_cycles +=1
        print("long cycle")
        print(difference_permutation)    

 # Count the cycles in difference_permutation
    for cycle in find_permutation.omit_singletons(difference_permutation):
        # Convert cycle to tuple for dictionary key
        print('cycle_decomposition ' + str(cycle_decomposition)  + ' at step ' + str(step))
        print(cycle)
        cycle_key = tuple(cycle)
        cycle_counts[cycle_key] += 1

    # increase counters for cycles in difference_permutation 
    simplified_permuation = find_permutation.omit_singletons(difference_permutation)
    cycle_length_sum = find_permutation.cycle_length_sum(simplified_permuation)
    collisions += cycle_length_sum

        # Update maximum cycle_length_sum and corresponding step
    if cycle_length_sum > max_cycle_length_sum:
        max_cycle_length_sum = cycle_length_sum
        max_cycle_length_sum_step = step
        store_difference_permutation = simplified_permuation

print()    

print("collisions")
print(collisions)

print("first collision at step " + str(first_collision_step))
print("last collision at step " + str(last_collision_step))
print("step with largest cycle_length_sum:", max_cycle_length_sum_step)
print("largest cycle_length_sum:", max_cycle_length_sum)
print("largest cycle_length_sum:", store_difference_permutation)
print("total steps with long cycle difference:", total_steps_with_long_cycles)

# Sort the cycles by frequency (value) in descending order
sorted_cycle_counts = sorted(cycle_counts.items(), key=lambda item: item[1], reverse=True)

# Print the sorted cycles with their counts
# print("Sorted Cycle Counts (by frequency):")
# for cycle, count in sorted_cycle_counts:
#     print(f"{cycle}: {count}")


