Main
1. initalize an eigenvalue summary with main_s_data.py
2. compute rotation process at a given s_step using main_t_data.py
3. run animation.py for visualization of the rotating process.

ToDo/Fix
1. Compute some more relevant data for summary/animation such as: estimate collision values (i,j,s,t) and smallest distances (summary), Delaunay triangulations (animation).
2. Method to compute all collisions should insert a s_step in between consecutive steps that differ by a permutation with some cycle > 3 (so that collisions are properly accounted for).