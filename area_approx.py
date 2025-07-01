import os
import json
import ctypes
import numpy as np
import pandas as pd
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

def monte_carlo_estimate (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), n_pts=10**6):
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
    
    # Approximate area of the lemniscate
    area = total_area * (inside_points / n_pts)
    stdev = total_area * np.sqrt((inside_points / n_pts) * (1 - inside_points / n_pts) / n_pts)

    return area, stdev

# C++ Monte Carlo implementation
lib = ctypes.CDLL("poly_lemniscate_project/libmontecarlo.so")
lib.monte_carlo_estimate.restype = ctypes.c_double
lib.monte_carlo_estimate.argtypes = [
    ctypes.POINTER (ctypes.c_double),  # pointer to roots_re 
    ctypes.POINTER (ctypes.c_double),  # pointer to roots_im
    ctypes.c_int,                      # degree
    ctypes.c_double, ctypes.c_double,  # xmin, xmax
    ctypes.c_double, ctypes.c_double,  # ymin, ymax
    ctypes.c_int                       # n_pts
]

def monte_carlo_estimate_cpp (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), n_pts=10**6):
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
        n_pts
    )
    
    return area

# C++ Hybrid Adaptive Mesh Refinement implementation

lib_amr = ctypes.CDLL("poly_lemniscate_project/libamr.so")
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


if __name__ == "__main__":
    filepath = "poly_lemniscate_project/Samples"
    filename = "samples_0628-0045"
    samples = load_viewing_samples(os.path.join(filepath, f"{filename}.json"))
    starting_degree = len(samples[0][0])  # Degree of the first sample


    for n in [1e4, 6.25e4, 2.5e5, 1e6, 4e6, 1e7]:

        areas = {}
        degrees = range(starting_degree, starting_degree + 9, 2)

        t = time.time()
        for i, deg_samples in zip(degrees, samples):
            areas[i] = []
            for j, roots in enumerate(deg_samples):
                # area = grid_estimate(roots, res=res)
                # area = monte_carlo_estimate(roots, n_pts=int(n))
                # area = monte_carlo_estimate_cpp(roots, n_pts=int(n))
                area = hybrid_amr_cpp(roots, max_depth=11, min_cell_size=s)
                areas[i].append(area)
        t = time.time() - t
        print(f"Total time: {t:.4f}s || Average time per lemniscate : {t / 30:.4f}s")

        df = pd.DataFrame(areas).T
        df.to_csv(os.path.join(filepath, f"{filename}_{n}pts_cpp.csv"), index=True, header=False)
        print(f"Areas saved")

# From initial testing, at the same number of points, Monte Carlo seems slower than grid estimation, but it is proably more accurate.
# In theory, we know the uncertainty of the Monte Carlo estimate, and with a million points, it was found to be ~0.3% and takes about 0.06-0.07s
# However in theory, the uncertainty with the grid estimate increases with the resolution, so I don't find it feasible.
# Especially assuming that our transformer will be very sensitive to the area of the lemniscates