import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random_matrix_model import points_on_curve
import random_matrix_model.initial_matrix_writter as initial_matrix_writter
import random_matrix_model.points_on_curve as curves
import computation.random_matrix_model as rmmodel

# Define parameter values
dim = 10
distribution = 'bernoulli'
remove_trace = True
curve = curves.Curve.CIRCLE
seed = 1070

# Generate initial matrix C
initial_matrix_type = initial_matrix_writter.generate_initial_matrix(dim, distribution, remove_trace, seed, curve)
initial_matrix = initial_matrix_type['matrix']

# Define animation parameters
s_0, s_1 = 0.00001, 1
s_step = 0.0001
t_steps = 100
s_increase = s_step / t_steps
s = s_0

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter([], [], color='blue', alpha=0.6, label="Current Eigenvalues")
sc_prev = ax.scatter([], [], color='red', alpha=0.6, label="Previous Eigenvalues")

ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_title("Eigenvalues Animation")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Generate frames
def update(frame):
    t = (frame % t_steps) / t_steps
    s = s_0 + frame * s_increase

    print(f"t={t}, s={s}")

    eigenvalues = np.linalg.eigvals(rmmodel.simple_generate_ginibre_whirlwind(initial_matrix, s, t, curve))

    if frame > t_steps:
        t_prev = ((frame - t_steps) % t_steps) / t_steps
        s_prev = s_0 + (frame - t_steps) * s_increase
        eigenvalues_prev = np.linalg.eigvals(rmmodel.simple_generate_ginibre_whirlwind(initial_matrix, s_prev, t_prev, curve))
        sc_prev.set_offsets(np.column_stack((np.real(eigenvalues_prev), np.imag(eigenvalues_prev))))
    else:
        sc_prev.set_offsets(np.empty((0, 2)))  # Properly hide red points initially

    sc.set_offsets(np.column_stack((np.real(eigenvalues), np.imag(eigenvalues))))
    ax.set_title(f"Eigenvalues Animation (s={s:.2f}, t={t:.2f})")

    return sc, sc_prev

# Create animation
ani = animation.FuncAnimation(fig, update, frames=int((s_1 - s_0) / s_step) * t_steps, interval=50, blit=True)

plt.show()
