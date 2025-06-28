# import numpy as np
# import matplotlib.pyplot as plt

# def main():
#     np.random.seed(42)
#     N = 100  # matrix size
#     a = 0.05  # interpolation parameter in [0,1]

#     # Derived correlation parameter
#     rho = 2 * np.sqrt(a * (1 - a))

#     # Complex Ginibre matrix with correct normalization
#     C = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2 * N)

#     # Interpolated matrix: sqrt(1 - a) * C + sqrt(a) * C*
#     A = np.sqrt(1 - a) * C + np.sqrt(a) * C.conj().T

#     # Eigenvalues
#     eigvals = np.linalg.eigvals(A)

#     # Theoretical ellipse boundary for given rho
#     theta = np.linspace(0, 2 * np.pi, 500)
#     ellipse = (1 + rho) * np.cos(theta) + 1j * (1 - rho) * np.sin(theta)


#     # Match style from the first script
#     plt.figure(figsize=(6, 3))
#     plt.plot(ellipse.real, ellipse.imag, 'r--', label=f'Ellipse (ρ = {rho:.3f})')
#     plt.scatter(eigvals.real, eigvals.imag, color="red", marker="o", edgecolor="black", s=10, alpha=0.5, label='Eigenvalues')

#     # Formatting
#     plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
#     plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
#     plt.xlabel("Re(λ)")
#     plt.ylabel("Im(λ)")
#     plt.grid(True, linestyle="--", linewidth=0.5)
#     plt.legend(loc="upper right")
#     plt.xlim(-2.1, 2.1)
#     plt.ylim(-1.1, 1.1)

#     # Keep x/y aspect ratio = 1
#     plt.gca().set_aspect('equal', adjustable='box')

#     plt.tight_layout()
#     plt.savefig("eigenvalues_plot.pdf", format="pdf", dpi=300, bbox_inches="tight")
#     plt.show()

# if __name__ == "__main__":
#     main()



# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# def animate_eigenvalues():
#     np.random.seed(42)
#     N = 100  # matrix size

#     # Precompute Ginibre matrix
#     C = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2 * N)

#     fig, ax = plt.subplots(figsize=(6, 3))
#     scat = ax.scatter([], [], color="red", marker="o", edgecolor="black", s=10, alpha=0.5)
#     ellipse_line, = ax.plot([], [], 'r--', linewidth=1)

#     ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
#     ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
#     ax.set_xlim(-2.1, 2.1)
#     ax.set_ylim(-2.1, 2.1)
#     ax.set_xlabel("Re(λ)")
#     ax.set_ylabel("Im(λ)")
#     ax.set_aspect('equal', adjustable='box')
#     ax.grid(True, linestyle="--", linewidth=0.5)
#     ax.legend(["Ellipse boundary", "Eigenvalues"], loc="upper right")

#     theta = np.linspace(0, 2 * np.pi, 500)

#     def update(frame):
#         x = 2 * np.pi * frame / num_frames  # full spin
#         rho = np.sin(2 * x)
#         A = np.cos(x) * C + np.sin(x) * C.conj().T
#         eigvals = np.linalg.eigvals(A)

#         ellipse = (1 + rho) * np.cos(theta) + 1j * (1 - rho) * np.sin(theta)

#         scat.set_offsets(np.c_[eigvals.real, eigvals.imag])
#         ellipse_line.set_data(ellipse.real, ellipse.imag)
#         ax.set_title(f"x = {x:.2f} rad  |  ρ = {rho:.2f}")
#         return scat, ellipse_line

#     num_frames = 1000
#     ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     animate_eigenvalues()
