import numpy as np
from random_matrix_model import points_on_curve

def create_initial_matrix_dtype(dim):
    initial_matrix_dtype = np.dtype([
        ('matrix', np.complex128, (dim, dim)),  # dim x dim complex matrix
        ('seed', np.integer),  # seed number (for reproducibility)
        ('curve', points_on_curve.Curve),  # Curve enum type
        ('properties', np.unicode_, 100)  # String properties
    ])
    return initial_matrix_dtype

def create_summary_item_dtype(dim):
    summary_item_dtype = np.dtype([
        ('t', np.float64),  # Time parameter
        ('s', np.float64),  # Additional scalar parameter
        ('eigenvalues', np.complex128, (dim,)),  # dim eigenvalues
        ('ordered', np.bool_),  # Whether eigenvalues are ordered
        ('associated_permutation', np.integer, (dim,))  # Permutation of eigenvalues
    ])
    return summary_item_dtype

def create_ginibre_summary_dtype(dim, m):
    initial_matrix_dtype = create_initial_matrix_dtype(dim)
    summary_item_dtype = create_summary_item_dtype(dim)

    ginibre_summary_dtype = np.dtype([
        ('initial_matrix', initial_matrix_dtype),  # Initial matrix information
        ('summary_items', summary_item_dtype, (m,))  # Array of m summary_items
    ])

    return ginibre_summary_dtype

def create_tracking_data_summary_dtype(dim, m):
    initial_matrix_dtype = create_initial_matrix_dtype(dim)
    summary_item_dtype = create_summary_item_dtype(dim)

    ginibre_summary_dtype = np.dtype([
        ('initial_matrix', initial_matrix_dtype),  # Initial matrix information
        ('summary_items', summary_item_dtype, (m,))  # Array of m summary_items
    ])

    return ginibre_summary_dtype


# Example usage:
dim = 4  # Dimension of the matrix
m = 10   # Number of summary items
ginibre_summary_dtype = create_ginibre_summary_dtype(dim, m)
print(ginibre_summary_dtype)
