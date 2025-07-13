import os
import json
import ctypes
import time
import numpy as np
import pandas as pd
import seaborn as sns
from utils import *

def grid_estimate (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), res=1000):
    """
    Approximate the area of the lemniscate defined by its roots.

    Parameters:
    - roots: List of complex roots of the polynomial.
    - xlim, ylim: The limits for the x and y axes.
    - res: Resolution of the grid for approximation.

    Returns:
    - area: Approximate area of the lemniscate.
    """
    X, Y, mod_P = grid_evaluate_pol(roots, xlim=xlim, ylim=ylim, res=res)
    
    # Count points inside the lemniscate
    inside_points = np.sum(mod_P < 1.0)  # Assuming bound is 1.0
    
    dx = (xlim[1] - xlim[0]) / res
    dy = (ylim[1] - ylim[0]) / res
    pixel_area = dx * dy  
    
    # Approximate area of the lemniscate
    return inside_points * pixel_area



def recursive_refine (p, min_cell_size:float, depth:int, max_depth:int, xlim=(-2, 2), ylim=(-2, 2)):

    test_pts = [(xlim[0], ylim[0]), (xlim[1], ylim[0]),
               (xlim[0], ylim[1]), (xlim[1], ylim[1]),
                ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2)]
    test_pts = [complex(*pt) for pt in test_pts]
    in_count = sum(1 for pt in test_pts if np.abs(p(pt)) < 1.0)
    area = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])

    if in_count == 5:
        return area
    if in_count == 0 and depth > 0:
        return 0
    if depth >= max_depth or cell_size < min_cell_size:
        return area * (in_count / 5)

    x_mid, y_mid = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2

    area1 = recursive_refine(roots, min_cell_size, depth + 1, max_depth, (xlim[0], x_mid), (ylim[0], y_mid))
    area2 = recursive_refine(roots, min_cell_size, depth + 1, max_depth, (xlim[0], x_mid), (y_mid, ylim[1]))
    area3 = recursive_refine(roots, min_cell_size, depth + 1, max_depth, (x_mid, xlim[1]), (ylim[0], y_mid))
    area4 = recursive_refine(roots, min_cell_size, depth + 1, max_depth, (x_mid, xlim[1]), (y_mid, ylim[1]))

    return area1 + area2 + area3 + area4
    
def hybrid_amr_py (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), init_divs=8, min_cell_size=1e-5, max_depth=11):
    """
    Hybrid AMR in python
    """
    p = mk_pol(roots)
    total_area = 0
    dx, dy = (xlim[1] - xlim[0]) / init_divs, (ylim[1] - ylim[0]) / init_divs
    for i in range(init_divs):
        for j in range(init_divs):
            x_min = xlim[0] + i * dx
            x_max = xlim[0] + (i + 1) * dx
            y_min = ylim[0] + j * dy
            y_max = ylim[0] + (j + 1) * dy
            
            total_area += recursive_refine(p, min_cell_size, 0, max_depth,
                                    (x_min, x_max), (y_min, y_max))

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

# C++ Hybrid Adaptive Mesh Refinement implementation

lib_amr = ctypes.CDLL("/home/databoss465/poly_lemniscate_project/libamr.so")
lib_amr.hybrid_amr_estimate.restype = ctypes.c_double
lib_amr.hybrid_amr_estimate.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # pointer to roots_re
    ctypes.POINTER(ctypes.c_double),  # pointer to roots_im
    ctypes.c_int,                     # degree
    ctypes.c_double, ctypes.c_double, # xmin, xmax
    ctypes.c_double, ctypes.c_double, # ymin, ymax
    ctypes.c_int,                     # init_divs
    ctypes.c_double,                  # min_cell_size
    ctypes.c_int                      # max_depth
]

def hybrid_amr_cpp (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), init_divs=8,
                     min_cell_size=1e-3, max_depth=5):
    """
    Estimate the area of the lemniscate using a hybrid adaptive mesh refinement technique with C++ implementation.
    This function does a pre-tiling, i.e. divides the area into `init_divs` squares, and then refines the mesh recursively
    based on corners and midpoint until the cell size is less than `min_cell_size` or the maximum depth is reached.
    """ 
    degree = len(roots)
    roots_re = np.array([r.real for r in roots], dtype=np.double)
    roots_im = np.array([r.imag for r in roots], dtype=np.double)

    area = lib_amr.hybrid_amr_estimate(
        roots_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        roots_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        degree, xlim[0], xlim[1], ylim[0], ylim[1],
        init_divs, min_cell_size, max_depth)

    return area