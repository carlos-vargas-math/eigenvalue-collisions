import numpy as np
import matplotlib.pyplot as plt


def plot_ordered_complex_points(points):
    """
    Plots a scatter plot of ordered complex points with labels.

    Parameters:
        points (np.array): A numpy array of complex numbers, ordered.
    """
    if not np.iscomplexobj(points):
        raise ValueError("Input array must contain complex numbers.")

    # Extract real and imaginary parts
    real_parts = points.real
    imag_parts = points.imag
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(real_parts, imag_parts, color='blue', alpha=0.7, label='Points')
    
    # Add labels for each point
    for idx, (x, y) in enumerate(zip(real_parts, imag_parts)):
        plt.text(x, y, str(idx), fontsize=9, ha='right', va='bottom', color='red')
    
    # Set plot labels and title
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title("Ordered Complex Points")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='upper right')
    plt.axis('equal')  # Ensure the aspect ratio is equal
    plt.show()


loaded_data = np.load("ordered_s_eigenvalues.npy", allow_pickle=True)
# loaded_data = np.load("well_ordered_summaries/complexGaussianN30.npy", allow_pickle=True)
points = loaded_data['eigenvalues'][0]
plot_ordered_complex_points(points)
