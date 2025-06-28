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

### Getting started

Please visit this page
[this page](https://carlos-vargas-math.github.io/eigenvalue-collisions/animation.html)
to get an instant glimpse of the
main objects that concern the methods in this package.

### One seed

1. Go to settings.py and specify the seed (for reproductibility) 
and other parameters in the model (dimension, curve, distribution, s_steps, t_steps, m_grid).
Set seed = seed_end to avoid calculating data that you won't need.

2. Run main_s_data.py to initialize a summary. 
This will order the eigenvalues from
R(0,0) to R(1,0), along the curve t=0 into s_steps.

3. Run main_t_data.py to compute the eigenvalue tracks for the s-steps of your choice. 
For each selected s, the algorithm will order the eigenvalues of
R(s,0) to R(s,1) into t_steps, to figure out the cycles/tracks.

4. Run grid_search_summary.py and compute the eigenvalue collisions. 

5. Once these tracks and collisions have been calculated, indicate the steps that will be displayed in animation_tracks.py and run it.

### Several seeds

1. Go to settings.py and specify the seed_end > seed 
and other parameters in the model (dimension, curve, distribution, s_steps, t_steps, m_grid).

2. Run grid_search_summary.py and compute the eigenvalue collisions. 

3. Run to read_grid_summary.py to get basic statistics about the eigenvalue collisions. 

## License
MIT License



