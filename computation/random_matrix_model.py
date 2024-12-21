import numpy as np
import random_matrix_model.points_on_curve as curves

def flush_ginibre_toilet(initial_matrix, alpha, beta, t, curve_type):
    dim = np.size(initial_matrix, 0)
    curve = curves.parametrized_curve(dim, curve_type, t*dim)    

    ginibre = alpha * initial_matrix
    for j in range(dim):
        ginibre[j][j] = beta * curve[j]

    return ginibre

def simple_flush_ginibre_toilet(initial_matrix, s, t, curve_type):
    theta = s*np.pi/2
    return flush_ginibre_toilet(initial_matrix, np.cos(theta), np.sin(theta), t, curve_type)