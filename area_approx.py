import os
import json
import numpy as np
import pandas as pd
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

if __name__ == "__main__":
    path = "poly_lemniscate_project/Samples/samples_0628-0045.json"
    samples = load_viewing_samples(path)
    starting_degree = len(samples[0][0])  # Degree of the first sample

    areas = {}

    for i, deg_samples in enumerate(samples):
        areas[i+starting_degree] = []
        for j, roots in enumerate(deg_samples):
            area = grid_approx(roots)
            areas[i+starting_degree].append(area)

    print(areas)

