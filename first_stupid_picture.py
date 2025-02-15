import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate a 400 x 400 complex Gaussian random matrix
n = 400
matrix = (np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))) / np.sqrt(2)

# Compute eigenvalues for different matrices
eigenvalues = np.linalg.eigvals(matrix)
real_part_eigenvalues = np.linalg.eigvals(matrix.real)
imaginary_part_eigenvalues = np.linalg.eigvals(1j * matrix.imag)
sign_real_eigenvalues = np.linalg.eigvals(np.sign(matrix.real))
sign_imag_eigenvalues = np.linalg.eigvals(1j * np.sign(matrix.imag))

# Generate 400 independent points within the unit disc
points = []
while len(points) < 400:
    x, y = np.random.uniform(-1, 1, size=2)
    if x**2 + y**2 <= 1:
        points.append((x, y))
points = np.array(points)

# Titles and data for plots
plot_titles = [
    "Eigenvalues of the Complex Matrix",
    "Eigenvalues of the Real Part of the Matrix",
    "Eigenvalues of the Imaginary Part of the Matrix",
    "Eigenvalues of the Sign Matrix (Real Part)",
    "Eigenvalues of the Sign Matrix (Imaginary Part)",
    "400 Points Inside the Unit Disc",
]
eigenvalue_sets = [
    eigenvalues,
    real_part_eigenvalues,
    imaginary_part_eigenvalues,
    sign_real_eigenvalues,
    sign_imag_eigenvalues,
    points,
]
file_formats = ["png", "pdf"]

# Create and save scatter plots
for i, (data, title) in enumerate(zip(eigenvalue_sets, plot_titles)):
    plt.figure(figsize=(6, 6))
    if i == 5:  # Plot for points in the unit disc
        plt.scatter(data[:, 0], data[:, 1], s=5, color='green', alpha=0.7)
    else:  # Plots for eigenvalues
        plt.scatter(data.real, data.imag, s=2, color='blue', alpha=0.7)
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.title(title)
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.gca().set_aspect('equal', adjustable='box')
    for fmt in file_formats:
        plt.savefig(f"plot_{i+1}.{fmt}", format=fmt, dpi=300)
    plt.close()

print("Plots saved as PNG and PDF files.")
