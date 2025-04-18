import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from enum import Enum
from settings_ellipse import SettingsEllipse
s = SettingsEllipse.a

class Curve(Enum):
    ZERO = 1
    CIRCLE = 2
    CIRCUIT = 3
    CROSSING = 4
    ELLIPSE = 5

def parametrized_curve(n, curve_type:Curve, shift): 
    z = np.zeros(n, dtype=complex)
    h = np.zeros(n, dtype=float)
    j = np.zeros(n, dtype=float)
    for t in range(n):
        h[t] = (t + shift)% n
        j[t] = h[t]*(4+np.pi)/n
        if (curve_type == Curve.ZERO):
            z[t] = complex(0, 0)
        if (curve_type == Curve.CIRCLE):
            z[t] = complex(np.cos(2*h[t]*np.pi/n), np.sin(2*h[t]*np.pi/n))
        if (curve_type == Curve.CIRCUIT):
            if h[t] < n/2:
                t_0 = h[t]
                z[t] = complex(np.cos(2*t_0*np.pi/n), np.sin(2*t_0*np.pi/n))
            elif h[t] < 2*n/3:
                t_0 = h[t] - n/2
                z[t] = -2/3 + complex(-np.cos(6*t_0*np.pi/n), - np.sin(6*t_0*np.pi/n))/3
            elif h[t] < 5*n/6:
                t_0 = h[t] - 2*n/3
                z[t] = complex(-np.cos(6*t_0*np.pi/n), np.sin(6*t_0*np.pi/n))/3
            elif h[t] <= n:
                t_0 = h[t] - 5*n/6
                z[t] = 2/3 + complex(-np.cos(6*t_0*np.pi/n), -np.sin(6*t_0*np.pi/n))/3
        if (curve_type == Curve.CROSSING):
            if j[t] < 2:
                t_0 = j[t]
                z[t] = complex(1-t_0, 0)
            elif j[t] < 2 + np.pi/2:
                t_0 = j[t] - 2
                z[t] = complex(-np.cos(t_0), np.sin(t_0))
            elif j[t] < 4 + np.pi/2:
                t_0 = j[t] - 2 - np.pi/2
                z[t] = complex(0, 1-t_0)
            elif j[t] < 4 + np.pi:
                t_0 = j[t] - 4 - np.pi/2
                z[t] = complex(np.cos(t_0+3*np.pi/2), np.sin(t_0+3*np.pi/2))
        if curve_type == Curve.ELLIPSE:
                rho = 2 * np.sqrt(s * (1 - s))
                a = 1 + rho
                b = 1 - rho
                n_dense = 5000

                # Dense theta grid for numerical integration
                theta_dense = np.linspace(0, 2 * np.pi, n_dense)
                dx = -a * np.sin(theta_dense)
                dy =  b * np.cos(theta_dense)
                ds = np.sqrt(dx**2 + dy**2)
                arc_lengths = cumulative_trapezoid(ds, theta_dense, initial=0)
                total_length = arc_lengths[-1]

                # Shift all arc length targets forward by a portion of total_length
                offset = (shift / n) * total_length
                target_arc_lengths = (np.linspace(0, total_length, n, endpoint=False) + offset) % total_length

                # Interpolate arc length to theta
                theta_from_s = interp1d(arc_lengths, theta_dense)
                theta_equidistant = theta_from_s(target_arc_lengths)

                for t in range(n):
                    theta = theta_equidistant[t]
                    x = a * np.cos(theta)
                    y = b * np.sin(theta)
                    z[t] = x + 1j * y

    return z

def diagonal_matrix(n, curve_type:Curve, shift):
    points = parametrized_curve(n, curve_type, shift)
    return np.diag(points)

# --- Animation of eigenvalues of diagonal matrices with shifted entries
n = 40
curve_type = Curve.ELLIPSE
num_frames = 300

# Generate consistent colors
colors = plt.cm.hsv(np.linspace(0, 1, n))

fig, ax = plt.subplots(figsize=(6, 6))
initial_z = parametrized_curve(n, curve_type, 0)
initial_eigvals = np.linalg.eigvals(np.diag(initial_z))
scat = ax.scatter(initial_eigvals.real, initial_eigvals.imag, s=30, c=colors)
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.set_title("Eigenvalues of Shifted Curve Diagonal Matrix")
ax.set_aspect('equal')

def update(frame):
    shift = (frame / num_frames) * n
    z = parametrized_curve(n, curve_type, shift)
    eigvals = np.linalg.eigvals(np.diag(z))
    scat.set_offsets(np.c_[eigvals.real, eigvals.imag])
    ax.set_title(f"Shift = {shift:.2f}")
    return scat,

ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
plt.show()