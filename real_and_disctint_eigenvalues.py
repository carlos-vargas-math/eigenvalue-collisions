import numpy as np
import random_matrix_model.initial_matrix_writter as initial_matrix_writter
import random_matrix_model.points_on_curve as curves

# Define parameter values
dim = 10
distribution = 'bernoulli'
remove_trace = True
curve = curves.Curve.CIRCLE

def count_real_eigenvalues(eigenvalues, tol=1e-7):
    """Count the number of real eigenvalues (imaginary part is close to zero)."""
    return np.sum(np.abs(np.imag(eigenvalues)) < tol)

def count_distinct_eigenvalues(eigenvalues, tol=1e-6):
    """Count the number of distinct eigenvalues, considering numerical precision."""
    distinct_vals = []
    for ev in eigenvalues:
        if not any(np.isclose(ev, dv, atol=tol) for dv in distinct_vals):
            distinct_vals.append(ev)
    return len(distinct_vals)

# Iterate over different seeds
for seed in range(1000, 1100):
    # Generate initial matrix C
    initial_matrix_type = initial_matrix_writter.generate_initial_matrix(dim, distribution, remove_trace, seed, curve)
    initial_matrix = initial_matrix_type['matrix']
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(initial_matrix)
    
    # Count real and distinct eigenvalues
    num_real = count_real_eigenvalues(eigenvalues)
    num_distinct = count_distinct_eigenvalues(eigenvalues)
    
    print(f"Seed: {seed}, Real eigenvalues: {num_real}, Distinct eigenvalues: {num_distinct}")
