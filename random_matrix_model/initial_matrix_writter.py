import numpy as np
import uuid
from random_matrix_model import points_on_curve
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datatypes1

def plot_eigenvalues(eigenvalues, title="Eigenvalues"):
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='blue', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.xlim(-1.1, 1.1)  # Set x-axis limits
    plt.ylim(-1.1, 1.1)  # Set y-axis limits    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


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


def generate_initial_matrix(dim, distribution, remove_trace=True, seed=None, curve=points_on_curve.Curve.CIRCLE):
    sigma = 1/np.sqrt(2*dim)
    # Create a random number generator
    rng = np.random.default_rng(seed)
    
    # Initialize properties string
    properties = f'distribution={distribution}'

    # Generate random complex numbers for the initial case
    s = np.array([complex(rng.normal(0, sigma), rng.normal(0, sigma)) for _ in range(dim * dim)])

    # Adjust the numbers for the specified distribution
    if distribution == 'realGaussian':
        properties = 'distribution=realGaussian'
        s = np.array([complex(np.sqrt(2)*x.real, 0) for x in s])  # Set imaginary part to zero
    elif distribution == 'bernoulli':
        properties = 'distribution=Bernoulli'
        s = np.array([complex(np.sqrt(1 / dim) * np.sign(x.real), 0) for x in s])  # Take sign of the real part

    # Reshape to a dim x dim matrix
    ginibre = s.reshape((dim, dim))
    if distribution == 'ginibreMeander':
        ginibre = generate_ginibre_meander(dim, seed)
        properties = ";distribution=ginibreMeander"  # Mark that it's a filtered matrix

    if distribution == 'opposingSectors':
        ginibre = generate_opposing_sectors(dim, seed)
        properties = ";distribution=opposingSectors"  # Mark that it's a filtered matrix

    # Remove the trace if required
    if remove_trace:
        np.fill_diagonal(ginibre, 0)
        properties += ";traceless=true"
    
    # Create the structured array using the initial_matrix dtype
    initial_matrix_dtype = datatypes1.create_initial_matrix_dtype(dim)
    result = np.zeros((), dtype=initial_matrix_dtype)
    result['matrix'] = ginibre
    result['seed'] = seed if seed is not None else -1  # Store the seed; -1 indicates no seed was set
    result['properties'] = properties
    result['curve'] = curve
    
    return result
