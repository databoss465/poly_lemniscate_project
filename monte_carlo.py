import os
import json
import ctypes
import time
import numpy as np
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
from utils import *

def monte_carlo_estimate_py (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), n_pts=10**6):
    """
    Estimate the area of the lemniscate using Monte Carlo method.

    Parameters:
    - roots: List of complex roots of the polynomial.
    - xlim, ylim: The limits for the x and y axes.
    - n_pts: Number of random samples to generate.

    Returns:
    - area: Estimated area of the lemniscate.
    """
    x = np.random.uniform(xlim[0], xlim[1], n_pts)
    y = np.random.uniform(ylim[0], ylim[1], n_pts)
    Z = x + 1j * y    
    p = mk_pol(roots)
    mod_P = np.abs(p(Z))
    
    # Count points inside the lemniscate
    inside_points = np.sum(mod_P < 1.0)  # Assuming bound is 1.0
    total_area = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])
    area = total_area * (inside_points / n_pts)

    return area

# C++ Monte Carlo implementation
lib = ctypes.CDLL("/home/databoss465/poly_lemniscate_project/libmontecarlo.so")
lib.monte_carlo_estimate.restype = ctypes.c_double
lib.monte_carlo_estimate.argtypes = [
    ctypes.POINTER (ctypes.c_double),  # pointer to roots_re 
    ctypes.POINTER (ctypes.c_double),  # pointer to roots_im
    ctypes.c_int,                      # degree
    ctypes.c_double, ctypes.c_double,  # xmin, xmax
    ctypes.c_double, ctypes.c_double,  # ymin, ymax
    ctypes.c_int,                      # n_pts
    ctypes.c_int                       # n_threads
]

def monte_carlo_estimate_cpp (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), n_pts=10**6, n_threads=2):
    """
    Estimate the area of the lemniscate using Monte Carlo method with C++ implementation.

    Parameters:
    - roots: List of complex roots of the polynomial.
    - xlim, ylim: The limits for the x and y axes.
    - n_pts: Number of random samples to generate.

    Returns:
    - area: Estimated area of the lemniscate.
    """
    degree = len(roots)
    roots_re = np.array([r.real for r in roots], dtype=np.double)
    roots_im = np.array([r.imag for r in roots], dtype=np.double)

    area = lib.monte_carlo_estimate(
        roots_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        roots_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        degree,
        xlim[0], xlim[1],
        ylim[0], ylim[1],
        n_pts, n_threads
    )
    
    return area