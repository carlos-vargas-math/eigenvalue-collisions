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

Go to settings.py and specify the seed (for reproductibility) 
and other parameters in the model (s_steps, t_steps).

1. Go to main_s_data to initialize a summary. This will order the eigenvalues from
R(0,0) to R(1,0), along the curve t=0 in s_steps.

2. Go to main_t_data to compute the eigenvalue tracks for several values of s_step. 
For each selected s, the algorithm will order the eigenvalues of
R(s,0) to R(s,1) in t_steps.

3. Go to grid_search_summary and compute the

4. Once these tracks have been calculated, indicate the steps that will be displayed in the animation (these must have been computed in the previous step).


### Several seeds



## License
MIT License



