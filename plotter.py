import os
import numpy as np
import matplotlib.pyplot as plt

def display_lemniscate (roots: list[complex], bound: float = 1.0, 
                        xlim: tuple = (-2, 2), ylim: tuple = (-2, 2), res: int = 1000,
                        path = "poly_lemniscate_project/Images/", count = 0):
    """
    Display the polynomial lemniscate defined by its roots.

    Parameters:
    - roots: List of complex roots of the polynomial.
    - bound: The boundary for the lemniscate, default set to 1.0
    - xlim, ylim: The limits for the x and y axes, default set to (-2, 2).
    - res: Resolution of the grid for plotting, default set to 1000.
    - path: Path to save the plot images, default set to "poly_lemniscate_project/Images/"
    - count: Counter for naming the saved plot images, default set to 0
    """
    #Create the polynomial from the roots
    coeffs = np.poly(roots)
    p = np.poly1d(coeffs)
    p_str = " + ".join([f"({c})x^{i}" if i > 0 else f"({c})" 
                 for i, c in zip(range(len(coeffs)-1, -1, -1), coeffs)])
    # print(f"Polynomial: " + p_str)

    # Create a grid of complex numbers
    x = np.linspace(xlim[0], xlim[1], res)
    y = np.linspace(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Evaluate the polynomial on the grid
    mod_P = np.abs(p(Z))
    # print(f"Min: {np.min(mod_P)}, Max: {np.max(mod_P)}")

    roots_re = [root.real for root in roots]
    roots_im = [root.imag for root in roots]

    root_info = root_info = "\n".join([f"{r.real:.3f} + {r.imag:.3f}j" for r in roots])

    # Plot the lemniscate
    plt.figure(figsize=(2*(xlim[1] - xlim[0]), 2*(ylim[1] - ylim[0])))
    plt.contour(X, Y, mod_P, levels=[bound], colors='blue')
    plt.scatter(roots_re, roots_im, color='red', label='Roots', s=25)

    plt.title(f"Lemniscate #{count}")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.axis('equal')
    plt.grid(True)

    plt.gca().text( 1.05, 0.95,
                    f"Roots:\n{root_info}",
                    transform=plt.gca().transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    plt.savefig(path + f"lemniscate_plot{count}.png", bbox_inches='tight')
    plt.close()

def root_generator (root_positions: list[tuple[float, float]]):
    """
    Since we restrict roots to lie on the unit circle, we can 
    genrate the roots taking one input parameter theta for each root
    """
    assert all(0 <= pos[0] <= 1 for pos in root_positions), "r values must be in the range [0, 1]"
    assert all(0 <= pos[1] <= 1 for pos in root_positions), "Theta values must be in the range [0, 1]"

    return [pos[0] * np.exp(2 * np.pi * 1j * pos[1]) for pos in root_positions]   

n = 5
degree = np.random.randint(7, 17)
print(degree)
for count in range(0, n):

    if count == 0:
        # Roots are evenly spaced on the unit circle
        # degree, r = 6, 1
        roots = [np.exp(2 * np.pi * 1j * k/ degree) for k in range(0, degree)]

    elif count == 1:
        # Roots are evenly spaced on the unit circle with a small random perturbation
        # degree, r = 6, 1
        del_theta = np.random.uniform(-0.005, 0.005, degree)
        roots = [np.exp((2 * np.pi * 1j * k / degree) + del_theta[k]) for k in range(0, degree)]

    elif count <= n//2:
        # Roots have a unfiormly random distribution on the unit circle
        # degree = np.random.randint(1, 25)
        r = np.random.uniform(0, 1, degree)
        theta = np.random.uniform(0, 1, degree)
        root_positions = list(zip(r, theta))
        roots = root_generator(root_positions)
    
    else:
        # Roots have a unfiromly random distribution in the unit disk
        # degree = np.random.randint(1, 25)
        r = [1] * degree
        theta = np.random.uniform(0, 1, degree)
        root_positions = list(zip(r, theta))
        roots = root_generator(root_positions)

    # print(degree)
    display_lemniscate(roots, count=(n+count))

print("Plots generated")


