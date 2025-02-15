from enum import Enum
import numpy as np

class Curve(Enum):
    ZERO = 1
    CIRCLE = 2
    CIRCUIT = 3
    CROSSING = 4

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
    return z

def diagonal_matrix(n, curve_type:Curve, shift):
    points = parametrized_curve(n, curve_type, shift)
    return np.diag(points)
