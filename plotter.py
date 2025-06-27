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
    print(f"Polynomial: " + p_str)

    # Create a grid of complex numbers
    x = np.linspace(xlim[0], xlim[1], res)
    y = np.linspace(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Evaluate the polynomial on the grid
    mod_P = np.abs(p(Z))
    print(f"Min: {np.min(mod_P)}, Max: {np.max(mod_P)}")

    # Plot the lemniscate
    plt.figure(figsize=(2*(xlim[1] - xlim[0]), 2*(ylim[1] - ylim[0])))
    plt.contour(X, Y, mod_P, levels=[bound], colors='blue')
    plt.title(f"{bound}-Lemniscate of {p_str}")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(path + f"lemniscate_plot{count}.png")
    plt.close()

def standard_lemniscate(n: int, start: int, end: int, step: int, count = 0):
    """
    Generate and display the lemniscate for roots on the unit circle
    separated by integral multiples of pi/n.

    Parameters:
    - n: multiplier for the space between roots
    - start: Starting index for the roots
    - end: Ending index for the roots
    - step: Step size for the roots
    """
    roots = [np.exp(1j * np.pi / n * k) for k in range(start, end, step)]
    display_lemniscate(roots, bound=1, count=count)
    return


# count = 5
# roots = [1, 1j, -1, -1j]
# display_lemniscate(roots, bound=1, count=count)

standard_lemniscate(4, 0, 8, 1, count=6)

