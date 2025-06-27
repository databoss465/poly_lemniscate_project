import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

def root_generator_disk (root_positions: list[tuple[float, float]]):
    """
    Generate roots that are uniformly distributed in the unit disk.
    Each root is represented by a tuple (r, theta)
    """
    assert all(0 <= pos[0] <= 1 for pos in root_positions), "r values must be in the range [0, 1]"
    assert all(0 <= pos[1] <= 1 for pos in root_positions), "Theta values must be in the range [0, 1]"

    return [pos[0] * np.exp(2 * np.pi * 1j * pos[1]) for pos in root_positions]

def root_generator_circle (root_positions: list[float]):
    """
    Generate roots that are uniformly distributed on the unit circle.
    """
    assert all(0 <= pos <= 1 for pos in root_positions), "Theta values must be in the range [0, 1]"
    
    return [np.exp(2 * np.pi * 1j * pos) for pos in root_positions]

def perturb_roots (roots, perturbation=0.005):
    """
    Apply a small random perturbation to the roots.
    """
    del_theta = np.random.uniform(-perturbation, perturbation, len(roots))
    return [root * np.exp(del_theta[k] * 1j) for k, root in enumerate(roots)]

def mk_pol (roots):
    """
    Generate a polynomial from the given roots.
    """
    coeffs = np.poly(roots)
    p = np.poly1d(coeffs)
    return p

def grid_evaluate_pol (roots: list[complex], xlim=(-2, 2), ylim=(-2, 2), res=1000):
    """
    Evaluate the polynomial on a grid of complex numbers.

    Parameters:
    - roots: List of complex roots.
    - xlim: Tuple specifying the x-axis limits.
    - ylim: Tuple specifying the y-axis limits.
    - res: Resolution of the grid.

    Returns:
    - X: 2D array of x-coordinates.
    - Y: 2D array of y-coordinates.
    - mod_P: 2D array of the modulus of the polynomial evaluated at the grid
    """
    x = np.linspace(xlim[0], xlim[1], res)
    y = np.linspace(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    p = mk_pol(roots)
    mod_P = np.abs(p(Z))
    return X, Y, mod_P

def generate_viewing_samples (deg:int, save=False):
    """
    This function generates samples for viewing and comparing the lemniscates
    Given a degree, it generates 6 samples, one evenly spaced, one perturbed
    two uniformly random on the unit circle, and two uniformly random in the unit disk.
    It also does this for deg +- 2 and deg +- 4, meant to be viewed on a 5x6 axes grid.
    """
    assert deg >= 6, "Degree must be at least 6 to generate samples."
    samples = []                            #List of 5 lists, each containing 6 samples of roots

    for d in range(deg - 4, deg + 5, 2):
        deg_samples = []                    #List of 6 lists, each containing roots for a specific sample

        for count in range(6):
            if count == 0:
                # Roots are evenly spaced on the unit circle
                roots = root_generator_circle([k / d for k in range(0, d)])
            elif count == 1:
                # Roots are evenly spaced on the unit circle with a small random perturbation
                roots = root_generator_circle([k / d for k in range(0, d)])
                roots = perturb_roots(roots)
            elif count <= 3:
                # Roots have a uniformly random distribution on the unit circle
                roots = root_generator_circle(np.random.uniform(0, 1, d))
            else:
                # Roots have a uniformly random distribution in the unit disk
                r = np.random.uniform(0, 1, d)
                theta = np.random.uniform(0, 1, d)
                root_positions = list(zip(r, theta))
                roots = root_generator_disk(root_positions)

            deg_samples.append(roots)
        samples.append(deg_samples)

    if save:
        sample_name = save_viewing_samples(samples)
    else:
        print("Samples generated but not saved.")

    return (samples, sample_name) if save else samples


def save_viewing_samples (samples, path="poly_lemniscate_project/Samples/"):
    """
    Save the generated viewing samples to a JSON file,using tuples to represent complex numbers as (real, imag) pairs.
    
    Parameters:
    - samples: List of samples to save.
    - path: Directory where the samples will be saved.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    timestamp = time.strftime("%m%d-%H%M")
    filename = f"samples_{timestamp}"
    filepath = os.path.join(path, f"{filename}.json")

    root_dict = {}
    start = len(samples[0][0])  # Degree of the first sample
    degs = range(start, start + 9, 2)
    for deg_idx, degree_samples in zip(degs, samples):
        root_dict[f"deg_{deg_idx}"] = []
        for sample in degree_samples:
            root_list = [(r.real, r.imag) for r in sample]
            root_dict[f"deg_{deg_idx}"].append(root_list)

    with open(filepath, 'w') as f:
        json.dump(root_dict, f, indent=4)  

    print(f"Samples saved to {filepath}")

    return filename

def load_viewing_samples (path):
    with open(path, 'r') as f:
        data = json.load(f)
        # print(data.keys())
        # print(data['deg_11'])

    samples = []
    for deg_key in data.keys():
        deg_samples = []
        for root_list in data[deg_key]:
            roots = []
            for root in root_list:
                roots.append(complex(root[0], root[1]))
            deg_samples.append(roots)
        samples.append(deg_samples)

    return samples

if __name__ == "__main__":
    # Example usage
    samples, name = generate_viewing_samples(15, save=True)
    print(f"Samples generated and saved as {name}")
    samples2 = generate_viewing_samples(10)




