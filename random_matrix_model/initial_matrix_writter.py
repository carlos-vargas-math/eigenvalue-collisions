import numpy as np
import uuid
from random_matrix_model import points_on_curve

import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datatypes1


def generate_initial_matrix(dim, distribution='complexGaussian', remove_trace=True, seed=None, curve=points_on_curve.Curve.CIRCLE):
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
        s = np.array([complex(x.real, 0) for x in s])  # Set imaginary part to zero
    elif distribution == 'bernoulli':
        properties = 'distribution=Bernoulli'
        s = np.array([complex(np.sqrt(1 / dim) * np.sign(x.real), 0) for x in s])  # Take sign of the real part

    # Reshape to a dim x dim matrix
    ginibre = s.reshape((dim, dim))

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
