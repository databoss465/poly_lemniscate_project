import os
import json
import ctypes
import time
import numpy as np
import pandas as pd
import seaborn as sns
from utils import *
from monte_carlo import *
from amr import *


def score (root_positions: list[float], **kwargs) -> float:
    """
    Calculate the score of the polynomial defined by its root positions on the unit circle.
    Since the objective is minimize the area, the score is `4 - area`.

    Keyword arguments:
    - method: monte_carlo, hybrid_amr, monte_carlo_py, hybrid_amr_py
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
    elif method == 'monte_carlo_cuda':
        area = monte_carlo_estimate_cuda(roots, n_pts=n_pts, n_threads=n_threads)
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
    """
        Keyword arguments:
    - method: monte_carlo, hybrid_amr, monte_carlo_py, hybrid_amr_py
    - n_pts: Number of points for Monte Carlo method (default: 10^6)
    - init_divs: Initial divisions for hybrid AMR (default: 8)
    - min_cell_size: Minimum cell size for hybrid AMR (default: 1e-5)
    - max_depth: Maximum depth for hybrid AMR (default: 11)
    """
    assert type in ['standard', 'scaling']
    path = f"Samples/{type}_benchmark.json"
    sample_set = json.load(open(path, 'r'))
    n_deg = len(sample_set)

    method = kwargs.get('method', 'default')
    print(f"Benchmarking with method: {method}")

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
    print(f"Normalized average runtime: {normalized_avg_runtime*1000:.6f}ms per degree")
    df = pd.DataFrame(rows)
    return df, normalized_avg_runtime
    


if __name__ == "__main__":
    bm_type = 'standard'
    methods = ['monte_carlo', 'hybrid_amr', 'monte_carlo_cuda']
    std_df = pd.DataFrame(columns=['degree', 'runtime', 'details'])
    navrts = []
    for method in methods:
        df, navrt = benchmark(type=bm_type, method=method)
        std_df = pd.concat([std_df, df], ignore_index=True)
        navrts.append(navrt)
    print(f"Standard Benchmarking completed")
    print(f"Normalized Average Runtimes: {navrts}")

    bm_type = 'scaling'
    scale_df = pd.DataFrame(columns=['degree', 'runtime', 'details'])
    for method in methods:
        df, navrt = benchmark(type=bm_type, method=method)
        scale_df = pd.concat([scale_df, df], ignore_index=True)
        navrts.append(navrt)
    print(f"Scaling Benchmarking completed")
    print(f"Normalized Average Runtimes: {navrts}")
    
    # Save the DataFrames to CSV files
    std_df.to_csv(f'Samples/{bm_type}_benchmarking_std.csv', index=False)
    scale_df.to_csv(f'Samples/{bm_type}_benchmarking_scale.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    sns.boxplot(data=std_df, x='degree', y='runtime', hue='details', ax=axes[0], showfliers=False)
    axes[0].set_title('Standard Benchmarking')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Runtime (s)')
    axes[0].legend(title='Details', loc='upper left')
    sns.boxplot(data=scale_df, x='degree', y='runtime', hue='details', ax=axes[1], showfliers=False)
    axes[1].set_title('Scaling Benchmarking')
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Runtime (s)')
    axes[1].legend(title='Details', loc='upper left')
    plt.tight_layout()
    plt.savefig(f'Images/Benchmarking.png', dpi=300)
    plt.close(fig)
