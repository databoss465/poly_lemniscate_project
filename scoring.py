import os
import json
import ctypes
import time
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

lib_amr = ctypes.CDLL("libamr.so")
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


def score (root_positions: list[float], **kwargs) -> float:
    """
    Calculate the score of the polynomial defined by its root positions on the unit circle.
    Since the objective is minimize the area, the score is `4 - area`.

    Keyword arguments:
    - method: Method to use for area approximation ('monte_carlo' or 'hybrid_amr')
    - n_pts: Number of points for Monte Carlo method (default: 10^6)
    - init_divs: Initial divisions for hybrid AMR (default: 8)
    - min_cell_size: Minimum cell size for hybrid AMR (default: 1e-5)
    - max_depth: Maximum depth for hybrid AMR (default: 11)
    """
    assert all(0 <= pos < 1 for pos in root_positions), "Theta values must be in the range [0, 1)"

    method = kwargs.get('method', 'hybrid_amr')
    n_pts = kwargs.get('n_pts', 10**6)
    init_divs = kwargs.get('init_divs', 8)
    min_cell_size = kwargs.get('min_cell_size', 1e-5)
    max_depth = kwargs.get('max_depth', 11)

    roots = root_generator_circle(root_positions)

    if method == 'monte_carlo':
        area = monte_carlo_estimate_cpp(roots, n_pts=n_pts)
    elif method == 'hybrid_amr':
        area = hybrid_amr_cpp(roots, init_divs=init_divs, min_cell_size=min_cell_size, max_depth=max_depth)
    else:
        raise ValueError(f"Unknown method: {method}")

    # return area
    return 4 - area

def score_file (path: str, savepath: str, precision: int = 16, **kwargs) -> float:
    """
    Decodes a file coming from makemore, and computes the scores.
    Saves a sorted csv file with two columns 'root_positions' and 'score'.
    """
    root_positions = file_decoder(path, precision) # List of root positions
    scores = []
    for pos in root_positions:
        try:
            scores.append(score(pos, **kwargs))
        except AssertionError:
            print(f"Invalid root positions: {pos}. Skipping...")
            scores.append(float('nan'))
    df = pd.DataFrame({'root_positions': root_positions, 'score': scores})
    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    after_drop = len(df)
    dropped_count = before_drop - after_drop
    print(f"Dropped {dropped_count} rows with invalid root positions.")
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    df.to_csv(savepath, index=False)

    print(f"Root positions and scores saved to {savepath}")

def standard_benchmark (func):
    path = "Samples/standard_benchmark.json"
    sample_set = json.load(open(path, 'r'))
    benchamrk_report = []
    for samples in sample_set:
        n, deg = len(samples), len(samples[0])
        runtimes = []
        for sample in smaples:
            t = time.time()
            score(sample, **kwargs)
            runtimes.append(time.time() - t)
        mean_runtime = np.mean(runtimes)
        median_runtime = np.median(runtimes)
        stddev_runtime = np.std(runtimes)
        max_runtime, min_runtime = np.max(runtimes), np.min(runtimes)
        benchamrk_report.append({
            'n_samples': n,
            'degree': deg,
            'mean' : mean_runtime,
            'median': median_runtime,
            'stddev': stddev_runtime,
            'max': max_runtime,
            'min': min_runtime
        })

    df = pd.DataFrame(benchamrk_report)

    return df
    


if __name__ == "__main__":

   df = standard_benchmark(score)
   print(df)