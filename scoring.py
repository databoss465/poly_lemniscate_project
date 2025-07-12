import os
import json
import ctypes
import time
import numpy as np
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
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
    n_threads = kwargs.get('n_threads', 2)

    roots = root_generator_circle(root_positions)

    # print(method)
    # print(method == "monte_carlo_py")

    if method == 'monte_carlo':
        area = monte_carlo_estimate_cpp(roots, n_pts=n_pts, n_threads=n_threads)
    elif method == 'hybrid_amr':
        area = hybrid_amr_cpp(roots, init_divs=init_divs, min_cell_size=min_cell_size, max_depth=max_depth)
    elif method == 'monte_carlo_py':
        area = monte_carlo_estimate_py(roots, n_pts=n_pts)
    elif method == 'hybrid_amr_py':
        area = hybrid_amr_py(roots, init_divs=init_divs, min_cell_size=min_cell_size, max_depth=max_depth)
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

def benchmark (type:str = 'standard', **kwargs):
    assert type in ['standard', 'scaling']
    path = f"Samples/{type}_benchmark.json"
    sample_set = json.load(open(path, 'r'))
    n_deg = len(sample_set)

    method = kwargs.get('method', 'default')
    # if method == 'monte_carlo':
    #     n_pts = kwargs.get('n_pts', 10**6)
    #     deets = f"{method}_n{n_pts}"
    # elif method == 'hybrid_amr':
    #     init_divs = kwargs.get('init_divs', 8)
    #     min_cell_size = kwargs.get('min_cell_size', 1e-5)
    #     max_depth = kwargs.get('max_depth', 11)
    #     deets = f"{method}_i{init_divs}_m{min_cell_size}_d{max_depth}"
    # elif method == 'monte_carlo_py':
    #     n_pts = kwargs.get('n_pts', 10**6)
    #     deets = f"{method}_n{n_pts}"
    # elif method == 'hybrid_amr_py':
    #     init_divs = kwargs.get('init_divs', 8)
    #     min_cell_size = kwargs.get('min_cell_size', 1e-5)
    #     max_depth = kwargs.get('max_depth', 11)
    #     deets = f"{method}_i{init_divs}_m{min_cell_size}_d{max_depth}"
    # else:
    #     raise ValueError(f"Unknown method: {method}")

    deets = f"{method}_" + "_".join(f"{k[0]}{v}" for k, v in kwargs.items() if k != 'method')

    rows = []
    normalized_avg_runtime = 0.0
    for samples in sample_set:
        n, deg = len(samples), len(samples[0])
        k = n
        print(f"Scoring {k} polynomials of degree {deg}...")
        i = 0
        for sample in samples:
            i += 1
            if i > k:
                break
            t = time.time()
            score(sample, **kwargs)
            t = time.time() - t
            normalized_avg_runtime += t / deg
            rows.append({
                'degree': deg,
                'runtime' : t,
                'details': deets
                })

    normalized_avg_runtime /= n_deg * k
    print(f"Normalized average runtime: {normalized_avg_runtime:.6f} seconds per polynomial per degree")
    df = pd.DataFrame(rows)
    return df, normalized_avg_runtime
    


if __name__ == "__main__":

   bm_type = 'scaling'  # or 'scaling'
   df, navrt = benchmark(bm_type, method='monte_carlo_py', n_pts=10**6, n_threads = 8)
#    print(df.head(25))
   sns.boxplot(df, x='degree', y='runtime', hue='details')
#    plt.suptitle('Standard Benchmarking', fontsize=16)
   plt.suptitle('Scale Benchmarking', fontsize=16)
   plt.title(f'Normalized Avg Runtime: {navrt*1000:.6f}ms', fontsize=12)
   plt.xlabel('deg')
   plt.ylabel('runtime (s)')
#    plt.savefig(f'Images/standard_benchmarking_mcpy.png', dpi=300)
   plt.savefig(f'Images/scale_benchmarking_mcpy.png', dpi=300)
   plt.close()