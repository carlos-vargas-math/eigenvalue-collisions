# Eigenvalue Collisions

This package locates eigenvalue collisions for t-periodic matrix-valued functions R(s, t). 
It tracks eigenvalues continuously along curves to detect eigenvalue collisions.

Make sure to have the following dependencies installed.

## Dependencies
- NumPy
- SciPy
- Matplotlib (optional, for visualization)

## Usage

The methods in this package can be used to compute eigenvalue collisions for a specific seed
or to collect statistical data about eigenvalue collisions for a range of seeds.

### Main Method

The main method (in grid_search_summary.py) computes the list of eigenvalue collisions: (s,t,lambda) together with useful data about the computation process: (number of detected collisions, number of squares of type ii, number of unprocessed squares, grid resolution)

Please visit
[this page](https://carlos-vargas-math.github.io/eigenvalue-collisions/animation.html)
to get an instant glimpse of the
main objects that concern the methods in this package.

### Getting started

1. Go to settings.py and specify the seed (for reproductibility) 
and other parameters in the model (dimension, curve, distribution, s_steps, t_steps, m_grid).

Set seed = seed_end if you only want the summary to compute
one case, or seed_end > seed if you want to compute several seeds.

2. Run grid_search_summary.py and compute the eigenvalue collisions. 

Run read_grid_summary_s_histogram.py (single seed) 
or read_grid_summary.py (several_seeds) for some basic statistics of the collisions.

### Visualization

3. Run main_s_data.py to initialize a summary. 
This will order the eigenvalues from
R(0,0) to R(1,0), along the curve t=0 into s_steps.

4. Run main_t_data.py to compute the eigenvalue tracks for the s-steps of your choice. 
For each selected s, the algorithm will order the eigenvalues of
R(s,0) to R(s,1) into t_steps, to figure out the cycles/tracks.


5. Once these tracks and collisions have been calculated, indicate the steps that will be displayed in animation_tracks_old.py and run it. It will display the animated eigenvalues with collored tracks, with all collisions happening between the first and the last track. 

## License
MIT License



