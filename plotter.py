import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *

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

    X, Y, mod_P = grid_evaluate_pol(roots, xlim=xlim, ylim=ylim, res=res)

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


def display_viewing_samples(samples, sample_name, xlim=(-2, 2), ylim=(-2, 2), res=1000, path="poly_lemniscate_project/Images/"):
    """
    Display the 5x6 viewing samples of lemniscates.

    Parameters:
    - samples: List of samples to display; must come from generate_viewing_samples.
    - xlim, ylim: The limits for the x and y axes.
    - res: Resolution of the grid for plotting.
    """
    assert len(samples) == 5, "Samples must contain 5 sets of lemniscates."
    
    for i, deg_samples in enumerate(samples):
        assert len(deg_samples) == 6, f"Each sample set must contain 6 lemniscates, found {len(deg_samples)} in set {i+1}."
        
    rows, cols = 5, 6
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    filepath = os.path.join(path, f"{sample_name}.png")

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j] 
            roots = samples[i][j]
            X, Y, mod_P = grid_evaluate_pol(roots, xlim=xlim, ylim=ylim, res=res)
            roots_re = [root.real for root in roots]
            roots_im = [root.imag for root in roots]
            # Plotting
            ax.contour(X, Y, mod_P, levels=[1.0], colors='blue', linewidths=0.5)
            ax.scatter(roots_re, roots_im, color='red', label='Roots', s=4)
            #Config
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
            ax.tick_params(labelbottom=False, labelleft=False)  # Hides tick labels
            ax.grid(True)
            ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    # samples, sample_name = generate_viewing_samples(15, save=True)
    # display_viewing_samples(samples, sample_name)
    deg, precision = 15, 16
    root_pos = np.random.uniform(0, 1, deg).tolist()
    root_pos = canonical(root_pos)
    roots = root_generator_circle(root_pos)
    bitstring = bitstring_encoder(root_pos, precision)
    print(f"Root positions: {root_pos}")
    print(f"Bitstring: {bitstring}")

    # i = np.random.randint(0, len(bitstring))
    # print(f"Index to modify: {i}")  
    # if bitstring[i] == '0':
    #     modified_bitstring = bitstring[:i] + '1' + bitstring[i+1:]
    # elif bitstring[i] == '1':
    #     modified_bitstring = bitstring[:i] + '0' + bitstring[i+1:]
    # else:
    #     modified_bitstring = bitstring[:i+1] + ('1' if bitstring[i] == '0' else '0') + bitstring[i+1:]
    # enc_dec_pos = bitstring_decoder(modified_bitstring, precision)

    enc_dec_pos = bitstring_decoder(bitstring, precision)
    enc_dec_pos = canonical(enc_dec_pos)
    enc_dec_roots = root_generator_circle(enc_dec_pos)
    # print(f"Modified bitstring: {modified_bitstring}")
    print(f"Decoded positions: {enc_dec_pos}")

    display_lemniscate(roots, count=84)
    display_lemniscate(enc_dec_roots, count=85)




    

