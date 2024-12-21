import numpy as np

def create_custom_dtype(dim):
    """
    Factory function to create a dtype with a specified dimension.
    """
    return np.dtype([
        ('t', np.float64),
        ('s', np.float64),
        ('eigenvalues', np.complex128, (dim,)),  # Store N (1D) eigenvalues for each matrix
        ('ordered', np.bool_),
        ('associated_permutation', np.integer, (dim,))  # Permutation of eigenvalues
    ])