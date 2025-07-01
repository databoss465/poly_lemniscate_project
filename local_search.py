import os
import time
import json
import numpy as np
from utils import *
from area_approx import *
from plotter import display_lemniscate

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


def local_search (root_positions: list[float], max_steps:int=100) -> list[complex]:
    """
    A simple algorithm to generate a new set of root positions so that
    the score is at least as good as the input root positions.
    """
    assert all(0 <= pos < 1 for pos in root_positions), "Theta values must be in the range [0, 1)"      # Initial score

    deg = len(root_positions)
    initial_score = score(root_positions)
    # print(f"Initial score: {initial_score}")
    new_positions = root_positions.copy()
    improved = False
    count = 0

    while not improved:
        count += 1
        i = np.random.randint(0, deg)
        delta = np.random.uniform(-0.05, 0.05)

        cddt_positions = new_positions.copy()
        cddt_positions[i] = (cddt_positions[i] + delta) % 1
        cddt_score = score(cddt_positions)


        if cddt_score >= initial_score:
            new_positions = cddt_positions
            # print(f"Intial score: {initial_score}, New score: {cddt_score}, Step: {count}, Changed position: {i}, Delta: {delta:.4f}")
            improved = True

        elif count > max_steps:
            print(f"Failed to improve after {count} iterations. Returning original positions.")
            break
              
    return new_positions, cddt_score, improved

def rep_local_search (root_positions: list[float], max_steps:int=100, reps:int=100, display_status:int=5, tolerance:int=10) -> list[complex]:
    """
    Run local search for a number of repetitions.
    """
    assert all(0 <= pos < 1 for pos in root_positions), "Theta values must be in the range [0, 1)"

    new_positions = root_positions.copy()
    failed_attempts = 0
    for i in range(reps):
        if i % display_status == 0:
            print(f"Iteration {i+1}/{reps}")
        new_positions, new_score, improved = local_search(new_positions, max_steps=max_steps)
        
        if not improved:
            failed_attempts += 1
            if failed_attempts >= tolerance:
                print(f"Failed to improve after {failed_attempts} consecutive attempts. Stopping local search.")
                break
        else:
            failed_attempts = 0
    
    return new_positions, new_score
        

if __name__ == "__main__":

    deg, reps = 20, 200
    # root_positions = [k / n for k in range(n)]
    root_positions = np.random.uniform(0, 1, deg).tolist()
    root_positions = canonical(root_positions)

    t = time.time()
    new_positions, new_score = rep_local_search(root_positions, max_steps=20, reps=reps)
    t = time.time() - t
    new_positions = canonical(new_positions)
    # new_roots = root_generator_circle(new_positions)
    # display_lemniscate(new_roots, count=69)
    init_score, fin_score = score(root_positions), score(new_positions)
    print(f"Initial score: {init_score:.4f}, Final score: {fin_score:.4f}, Time taken: {t:.2f}s")


    roots = root_generator_circle(root_positions)
    display_lemniscate(roots, count=70)

    new_roots = root_generator_circle(new_positions)
    display_lemniscate(new_roots, count=71)

# Prereq:
# 0. Generate starter dataset
#   0.1 Run x iterations of local search on random root positions
#   0.2 Rank the results by score and save top p%
#   0.3 Encode them as bitstrings and save to txt file
# 1. Set up makemore

# Workflow:
# 1. Train makemore (maybe using BPE)
# 2. Decode the output
# 3. Run local search on the decoded output
# 4. Rank the results by score and save top p%
# 5. Encode them as bitstrings and save to txt file
# 6. Repeat steps 1-5 until convergence or desired number of iterations

    

