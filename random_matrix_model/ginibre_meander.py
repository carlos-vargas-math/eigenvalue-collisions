import numpy as np
import matplotlib.pyplot as plt


def plot_eigenvalues(eigenvalues, title="Eigenvalues"):
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='blue', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def generate_ginibre_meander(dim, seed=None):
    rng = np.random.default_rng(seed)
    
    # Step 1: Generate a (dim * 18/10) x (dim * 18/10) complex Gaussian matrix
    dim_large = int(dim * 1.8)
    sigma = 1 / np.sqrt(2 * dim_large)

    G = rng.normal(0, sigma, (dim_large, dim_large)) + 1j * rng.normal(0, sigma, (dim_large, dim_large))
    
    # Step 2: Compute eigenvalues and filter them based on the given conditions
    eigenvalues = np.linalg.eigvals(G)
    plot_eigenvalues(eigenvalues)
    selected_eigenvalues = [
        z for z in eigenvalues
        if (z.imag > 0 and abs(z) > 1/3)  # Upper semicircle of radius 1
        or (z.imag < 0 and (abs(z + 2/3) < 1/3  # Left semicircle
                            or abs(z - 2/3) < 1/3))  # Right semicircle
    ]
                                 
    plot_eigenvalues(selected_eigenvalues)
    
    # Step 3: Retry with different seeds until we get exactly 'dim' eigenvalues
    max_attempts = 100  # Avoid infinite loops
    attempts = 0
    while len(selected_eigenvalues) != dim and attempts < max_attempts:
        print(len(selected_eigenvalues))
        G = rng.normal(0, sigma, (dim_large, dim_large)) + 1j * rng.normal(0, sigma, (dim_large, dim_large))
        eigenvalues = np.linalg.eigvals(G)
        selected_eigenvalues = [
            z for z in eigenvalues
            if (z.imag > 0 and abs(z) > 1/3)  # Upper semicircle of radius 1
            or (z.imag < 0 and (abs(z + 2/3) < 1/3  # Left semicircle
                                or abs(z - 2/3) < 1/3))  # Right semicircle
        ]

        attempts += 1
    if len(selected_eigenvalues) != dim:
        raise ValueError(f"Failed to find exactly {dim} eigenvalues after {max_attempts} attempts.")
    
    # Step 4: Construct diagonal matrix D
    D = np.diag(selected_eigenvalues)
    
    # Step 5: Generate another dim x dim complex Gaussian matrix
    H = rng.normal(0, sigma, (dim, dim)) + 1j * rng.normal(0, sigma, (dim, dim))
    
    # Step 6: Compute its eigen-decomposition to get U and V
    U, _, Vh = np.linalg.svd(H)

    # Step 7: Return the transformed matrix UDV
    return U @ D @ Vh, selected_eigenvalues, eigenvalues


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


# Example usage
dim = 20
meander_matrix, selected_eigenvalues, original_eigenvalues = generate_ginibre_meander(dim, seed=70)

# Plot original Ginibre eigenvalues
plot_eigenvalues(original_eigenvalues, title="Original Ginibre Eigenvalues")

# Plot selected eigenvalues
plot_eigenvalues(selected_eigenvalues, title="Selected Eigenvalues for Ginibre Meander")
