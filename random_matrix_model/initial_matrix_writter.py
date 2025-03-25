import numpy as np
from random_matrix_model import points_on_curve
import sys
import os
from scipy.stats import unitary_group
import datatypes1
from enum import Enum

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Distribution(Enum):
    BERNOULLI = "bernoulli"
    COMPLEX_GAUSSIAN = "complexGaussian"
    REAL_GAUSSIAN = "realGaussian"
    GINIBRE_MEANDER = "ginibreMeander"
    OPPOSING_SECTORS = "opposingSectors"

def generate_opposing_sectors(dim, seed=None):
    """Generates a matrix with eigenvalues constrained to the upper-left or lower-right sectors."""
    rng = np.random.default_rng(1000 * seed)
    
    # Step 1: Generate a larger complex Ginibre matrix
    dim_large = int(dim * 2)
    sigma = 1 / np.sqrt(2 * dim_large)

    max_attempts = 100  # Retry limit
    attempts = 0
    selected_eigenvalues = []
    
    while len(selected_eigenvalues) != dim and attempts < max_attempts:
        G = rng.normal(0, sigma, (dim_large, dim_large)) + 1j * rng.normal(0, sigma, (dim_large, dim_large))
        eigenvalues = np.linalg.eigvals(G)
        
        selected_eigenvalues = [
            z for z in eigenvalues 
            if (z.real < 0 and z.imag > 0)  # Upper-left sector
            or (z.real > 0 and z.imag < 0)  # Lower-right sector
        ]
        
        attempts += 1
    
    if len(selected_eigenvalues) != dim:
        raise ValueError(f"Failed to find exactly {dim} eigenvalues after {max_attempts} attempts.")
    
    # Step 2: Construct diagonal matrix D
    D = np.diag(selected_eigenvalues)
    
    # Step 3: Generate Haar-random unitary matrix
    U = unitary_group.rvs(dim, random_state=seed)
    
    # Step 4: Compute transformed matrix UDV†
    transformed_matrix = U @ D @ U.conj().T
    
    return transformed_matrix


def generate_ginibre_meander(dim, seed=None):
    """Generates a Ginibre meander matrix with a structured eigenvalue selection process."""
    rng = np.random.default_rng(1000 * seed)
    
    # Step 1: Generate a larger complex Ginibre matrix
    dim_large = int(dim * 1.8)
    sigma = 1 / np.sqrt(2 * dim_large)

    G = rng.normal(0, sigma, (dim_large, dim_large)) + 1j * rng.normal(0, sigma, (dim_large, dim_large))
    
    # Step 2: Compute eigenvalues and filter them
    eigenvalues = np.linalg.eigvals(G)
    selected_eigenvalues = [
        z for z in eigenvalues
        if (z.imag > 0 and abs(z) > 1/3)  # Upper semicircle of radius 1
        or (z.imag < 0 and (abs(z + 2/3) < 1/3  # Left semicircle
                            or abs(z - 2/3) < 1/3))  # Right semicircle
    ]
    
    # Step 3: Retry with different seeds until we get exactly 'dim' eigenvalues
    max_attempts = 100  
    attempts = 0
    while len(selected_eigenvalues) != dim and attempts < max_attempts:
        G = rng.normal(0, sigma, (dim_large, dim_large)) + 1j * rng.normal(0, sigma, (dim_large, dim_large))
        eigenvalues = np.linalg.eigvals(G)
        selected_eigenvalues = [
            z for z in eigenvalues
            if (z.imag > 0 and abs(z) > 1/3)
            or (z.imag < 0 and (abs(z + 2/3) < 1/3 or abs(z - 2/3) < 1/3))
        ]
        attempts += 1
    
    if len(selected_eigenvalues) != dim:
        raise ValueError(f"Failed to find exactly {dim} eigenvalues after {max_attempts} attempts.")

    # Step 4: Construct diagonal matrix D
    D = np.diag(selected_eigenvalues)

    # Step 5: Generate two Haar-random unitary matrices
    U = unitary_group.rvs(dim, random_state=seed)

    # Step 6: Compute transformed matrix UDV†
    transformed_matrix = U @ D @ U.conj().T  # This ensures eigenvalues stay the same
    return transformed_matrix

def generate_initial_matrix(dim, distribution: Distribution, remove_trace=True, seed=None, curve=points_on_curve.Curve.CIRCLE):
    sigma = 1/np.sqrt(2*dim)
    rng = np.random.default_rng(seed)
    
    properties = f'distribution={distribution.value}'
    
    s = np.array([complex(rng.normal(0, sigma), rng.normal(0, sigma)) for _ in range(dim * dim)])

    if distribution == Distribution.REAL_GAUSSIAN:
        s = np.array([complex(np.sqrt(2) * x.real, 0) for x in s])
    elif distribution == Distribution.BERNOULLI:
        s = np.array([complex(np.sqrt(1 / dim) * np.sign(x.real), 0) for x in s])
    
    ginibre = s.reshape((dim, dim))
    
    if distribution == Distribution.GINIBRE_MEANDER:
        ginibre = generate_ginibre_meander(dim, seed)
    elif distribution == Distribution.OPPOSING_SECTORS:
        ginibre = generate_opposing_sectors(dim, seed)
    
    if remove_trace:
        np.fill_diagonal(ginibre, 0)
        properties += ";traceless=true"
    
    initial_matrix_dtype = datatypes1.create_initial_matrix_dtype(dim)
    result = np.zeros((), dtype=initial_matrix_dtype)
    result['matrix'] = ginibre
    result['seed'] = seed if seed is not None else -1
    result['properties'] = properties
    result['curve'] = curve
    
    return result
