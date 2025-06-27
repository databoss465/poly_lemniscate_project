import numpy as np
from utils import *

def grid_approx (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), res=1000):
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
    total_points = res * res
    
    # Area of the bounding box
    area_box = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])
    
    # Approximate area of the lemniscate
    return (inside_points / total_points) * area_box