# Eigenvalue Collisions

This package analyzes eigenvalue collisions in a periodic matrix-valued function R(s, t). It tracks eigenvalues on curves to detect eigenvalue collisions.

Make sure to have the following dependencies installed

## Dependencies
- NumPy
- SciPy
- Matplotlib (optional, for visualization)

## Usage

The methods in this package can be used to compute eigenvalue collisions for a specific seed
or to collect statistical data about eigenvalue collisions for a range of seeds.

### One seed

1. Go to settings.py and specify the seed (for reproductibility) 
and other parameters in the model (dimension, curve, distribution, s_steps, t_steps, m_grid).

2. Go to main_s_data.py to initialize a summary. This will order the eigenvalues from
R(0,0) to R(1,0), along the curve t=0 in s_steps.

3. Go to main_t_data.py to compute the eigenvalue tracks for the s_step of your choice. 
For each selected s, the algorithm will order the eigenvalues of
R(s,0) to R(s,1) in t_steps, to figure out the cycles/tracks.

4. Go to grid_search_summary and compute the eigenvalue collisions. 

5. Once these tracks and collisions have been calculated, indicate the steps that will be displayed in animation_tracks.py

### Several seeds

1. Go to settings.py and specify the seed_end > seed 
and other parameters in the model (dimension, curve, distribution, s_steps, t_steps, m_grid).

2. Go to grid_search_summary and compute the eigenvalue collisions. 

3. Go to read_grid_summary to get basic statistics about the eigenvalue collisions. 

## License
MIT License



