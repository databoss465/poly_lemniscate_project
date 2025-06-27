import numpy as np
import matplotlib.pyplot as plt

def display_lemniscate (roots: list[complex], bound: float = 1.0, 
                        xlim: tuple = (-5, 5), ylim: tuple = (-5, 5), res: int = 500,
                        path = "/poly_lemniscate_project/", count = 0):
    """
    Display the polynomial lemniscate defined by its roots.

    Parameters:
    - roots: List of complex roots of the polynomial.
    - bound: The boundary for the lemniscate, default set to 1.0
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

    # Plot the lemniscate
    plt.figure(figsize=(2*(xlim[1] - xlim[0]), 2*(ylim[1] - ylim[0])))
    plt.contour(X, Y, mod_P, levels=[bound], colors='blue')
    plt.title(f"Lemniscate of the polynomial {p}")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.axis('equal') 
    plt.savefig(path + f"lemniscate_plot{count}.png")



roots = [1j, 2, 3j, 4]
display_lemniscate(roots)
