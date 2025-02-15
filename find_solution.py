import numpy as np
from scipy.optimize import root
import random_matrix_model.points_on_curve as points_on_curve
import random_matrix_model.initial_matrix_writter as initial_matrix_writter

import numpy as np

def R(s, t, C, U):
    """
    Matrix-valued function R(s, t) depending on s, t, C, and U.
    """
    return np.cos(s) * C + np.sin(s) * np.exp(2j * np.pi * t) * U

def R_shifted(s, t, C, U, alpha=2.0):
    """
    Shifted matrix-valued function R(s, t).
    """
    return R(s, t, C, U) + alpha * np.eye(C.shape[0])

def det_function_shifted(variables, C, U, target_lambda, alpha=2.0):
    """
    Computes the determinant and its partial derivatives with a shift.
    """
    s, t = variables
    mat = R_shifted(s, t, C, U, alpha) - (target_lambda + alpha) * np.eye(C.shape[0])
    
    # Determinant of the shifted matrix
    det_value = np.linalg.det(mat)
    
    # Derivatives of R with respect to s and t
    dR_ds = -np.sin(s) * C + np.cos(s) * np.exp(2j * np.pi * t) * U
    dR_dt = 2j * np.pi * np.sin(s) * np.exp(2j * np.pi * t) * U
    
    # Partial derivatives of determinant with respect to s and t
    adj_mat = np.linalg.inv(mat).T * det_value  # Adjugate matrix scaled by determinant
    d_det_ds = np.trace(adj_mat @ dR_ds)
    d_det_dt = np.trace(adj_mat @ dR_dt)
    
    return np.array([det_value.real, d_det_ds.real, d_det_dt.real])

def find_collision_shifted(C, U, s0, t0, target_lambda, alpha=2.0):
    """
    Finds the eigenvalue collision point (s, t) using a root-finding method with a shift.

    Parameters:
        C (ndarray): The constant matrix in the periodic function R(s, t).
        U (ndarray): The matrix used in the computation of R(s, t).
        s0 (float): Initial guess for s.
        t0 (float): Initial guess for t.
        target_lambda (complex): The target eigenvalue to match.
        alpha (float): Scalar to control the shift (default is 2.0).

    Returns:
        ndarray: The solution (s, t) where the eigenvalue collision occurs.

    Raises:
        ValueError: If root finding fails.
    """
    def equations(variables):
        s, t = variables
        # Compute R(s, t) with the shift
        R_st = R(s, t, C, U) + alpha * np.eye(C.shape[0])  # Adjust as needed for your R function
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(R_st)
        # Find the differences to the target lambda
        f1 = np.real(eigenvalues[0]) - np.real(target_lambda)
        f2 = np.imag(eigenvalues[0]) - np.imag(target_lambda)
        return [f1, f2]

    # Solve the equations using scipy.optimize.root
    result = root(equations, [s0, t0], method='hybr')
    
    if result.success:
        return result.x  # Return the solution (s, t)
    else:
        raise ValueError(f"Root finding failed: {result.message}")
