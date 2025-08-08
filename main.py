import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool
from tqdm import tqdm

from utils import *
from scoring import *
from plotter import display_lemniscate


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

def local_search (root_positions: list[float], max_steps:int=100, score_kwargs:dict={}) -> list[complex]:
    """
    A simple algorithm to generate a new set of root positions so that
    the score is at least as good as the input root positions. For kwargs check scoring.py.

    Returns the new root config, the new score and a boolean indicating improvement.
    """
    assert all(0 <= pos < 1 for pos in root_positions), "Theta values must be in the range [0, 1)"      # Initial score

    deg = len(root_positions)
    initial_score = score(root_positions, **score_kwargs)  # Initial score
    # print(f"Initial score: {initial_score}")
    new_positions = root_positions.copy()
    new_score = initial_score
    improved = False
    step = 0
    score_count = 0

    while not improved:
        step += 1
        i = np.random.randint(0, deg)
        delta = np.random.uniform(-0.05, 0.05)

        cddt_positions = new_positions.copy()
        cddt_positions[i] = (cddt_positions[i] + delta) % 1
        cddt_score = score(cddt_positions, **score_kwargs)  
        score_count += 1


        if cddt_score >= initial_score:
            new_positions = cddt_positions
            new_score = cddt_score
            # print(f"Intial score: {initial_score}, New score: {new_score}, Step: {step}, Changed position: {i}, Delta: {delta:.4f}")
            improved = True

        elif step > max_steps:
            # print(f"Failed to improve after {step} iterations. Returning original positions.")
            break
              
    return new_positions, new_score, improved, score_count

def rep_local_search (root_positions: list[float], max_steps:int=100, reps:int=100, display_status:int=5, tolerance:int=10, score_kwargs:dict={}) -> list[complex]:
    """
    Run local search for a number of repetitions.
    """
    assert all(0 <= pos < 1 for pos in root_positions), "Theta values must be in the range [0, 1)"

    new_positions = root_positions.copy()
    failed_attempts = 0
    total_score_count = 0
    for i in range(reps):
        new_positions, new_score, improved, score_count = local_search(new_positions, max_steps, score_kwargs)
        total_score_count += score_count
        # if i % display_status == 0:
        #     print(f"Iteration {i+1}/{reps}: New score: {new_score:.4f}")
        
        if not improved:
            failed_attempts += 1
            if failed_attempts >= tolerance:
                # print(f"Failed to improve after {failed_attempts} consecutive attempts. Stopping local search.")
                break
        else:
            failed_attempts = 0
    
    # print (f"We called score {total_score_count} times in total")
    return canonical(new_positions), new_score

# Worker function for parallel processing
def populator (bs:int, deg: int, max_steps:int=100, reps:int=100, tolerance:int=10, score_kwargs:dict={})  -> list[tuple[list[float], float]]:
    """
    Worker function for parallel processing.
    Generates a random set of root positions and runs local search on them.
    Returns a list of tuples containing the root positions and the score.
    """
    results = []
    np.random.seed(os.getpid())  # Ensure different seed for each worker
    # improvement_count = 0

    for _ in tqdm(range(bs), desc="Worker progress", position=0, leave=False):
        root_positions = canonical(np.random.uniform(0, 1, deg).tolist())
        # scr = score(root_positions, **score_kwargs)
        new_positions, new_score = rep_local_search(root_positions, max_steps=max_steps, reps=reps, tolerance=tolerance, score_kwargs=score_kwargs)
        # new_positions, new_score, improved, _ = local_search(root_positions, 10, score_kwargs=score_kwargs)
        results.append((new_positions, new_score))
        # if improved:
        #     improvement_count += 1
        # results.append((root_positions, scr))
    # print(f'Improved {improvement_count}/{bs}')
    return results



def generate_population (pop_size:int, deg:int, max_steps:int=100, reps:int=100, tolerance:int=10, score_kwargs:dict={}, num_workers: int = 12) -> pd.DataFrame:
    """
    Generate a sorted population of canonical root positions using parallel processing.

    Parameters:
    - pop_size: Total number of root configs to generate
    - deg: Degree of the polynomial
    - max_steps: Maximum number of times for local search to try and improve the score
    - reps: Number of repetitions of the local search to be applied
    - tolerance: Number of consecutive failed attempts before stopping local search
    - num_workers: Number of parallel workers to use
    """

    bs = pop_size // num_workers
    args = [(bs, deg, max_steps, reps, tolerance, score_kwargs) for _ in range(num_workers)]

    print(f"Generating population of {pop_size} root positions with degree {deg} using {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
            all_batches = pool.starmap(populator, args)

    # Flatten the list of lists
    population = [item for batch in all_batches for item in batch]
    df = pd.DataFrame(population, columns=['root_positions', 'score'])
    df['root_positions'] = df['root_positions'].apply(lambda x: canonical(x))
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)

    return df
    
        

if __name__ == "__main__":

    deg = 15
    total_pop = 10001
    max_steps = 10
    reps = 10
    tolerance = reps/5
    score_kwargs = {'method':'hybrid_amr', 'init_divs': 10, 'min_cell_size':1e-6, 'max_depth':10}
    # NOTE : 10001 samples of deg 15, took 1597s with monte_carlo and 812s with hybrid amr
    # score_kwargs = {'method': 'monte_carlo'}

    filepath = "Samples"
    filename = f"population{total_pop}_deg{deg}_dec.csv"
    plotpath = "Images"
    plotname = f"population{total_pop}_deg{deg}_dec.png"

    # population_df = pd.read_csv(os.path.join(filepath, filename))
    
    t = time.time()
    #Generate the population
    population_df = generate_population(total_pop, deg, max_steps=max_steps, reps=reps, tolerance=tolerance, score_kwargs=score_kwargs, num_workers=8)
    #Check distinct values
    pct = population_df['root_positions'].apply(tuple).nunique()/len(population_df)*100
    print(f'{pct:.3f}% Distinct values')
    #Save CSV
    population_df.to_csv(os.path.join(filepath, filename), index=False)
    t = time.time() - t
    print(f"Population generated and saved to {os.path.join(filepath, filename)} in {t:.2f}s.")

    plt.hist(population_df['score'], bins=20, color='cyan', alpha=0.7, edgecolor='black')
    plt.title(f"Score Distribution for Degree {deg} Population, generated by Makemore")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plotpath, plotname), dpi=300)
    plt.close()






