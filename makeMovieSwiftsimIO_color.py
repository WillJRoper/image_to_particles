"""
Makes a movie of the KH 2D data.

You will need to run your movie with far higher time-resolution than usual to
get a nice movie; around 450 snapshots over 6s is required.

Edit this file near the bottom with the number of snaps you have.

Written by Josh Borrow (joshua.borrow@durham.ac.uk)
"""
import sys
import os
import h5py as h5
import numpy as np
import scipy.interpolate as si
import unyt

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid


def load_and_extract(filename):
    """
    Load the data and extract relevant info.
    """

    return load(filename)


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def make_plot(filename, array, nx, ny, dx, dy):
    """
    Load the data and plop it on the grid using nearest
    neighbour searching for finding the 'correct' value of
    the density.
    """

    data = load_and_extract(filename)

    mesh = project_gas_pixel_grid(data, nx, backend="fast")

    array.set_array(mesh)

    return (array,)


def frame(n, *args):
    """
    Make a single frame. Requires the global variables plot and dpi.
    """

    global plot, dpi

    fn = "{}_{:04d}.hdf5".format(filename, n)

    return make_plot(fn, plot, dpi, dpi, (0, 1), (0, 1))


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    from tqdm import tqdm
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LogNorm

    import swiftascmaps
    import matplotlib.pyplot as plt

    filename = sys.argv[1]
    dpi = int(sys.argv[2])
    outfilename = sys.argv[3]

    # Load colordata
    rgbs = np.load("rgbs.npy")

    # Look for the number of files in the directory.
    i = 0
    while True:
        if os.path.isfile("{}_{:04d}.hdf5".format(filename, i)):
            i += 1
        else:
            break

        if i > 10000:
            raise FileNotFoundError(
                "Could not find the snapshots in the directory")

    frames = tqdm(np.arange(0, i))

    for n in frames:

        fn = "{}_{:04d}.hdf5".format(filename, n)

        data = load_and_extract(fn)

        # Creation of first frame
        fig, ax = plt.subplots(1, 1, figsize=(1, 1), frameon=False)
        ax.axis("off")  # Remove annoying black frame.

        mesh = project_gas_pixel_grid(data, dpi)

        # Global variable for set_array
        plot = ax.imshow(
            mesh,
            extent=[0, 1, 0, 1],
            animated=True,
            interpolation="none",
            cmap="swift.midnights",
        )

        # Remove all whitespace
        fig.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=None, hspace=None)

        fig.savefig("{}_{:04d}_greyscale.png".format(outfilename, n), dpi=dpi)
        plt.close(fig)

        # Creation of first frame
        fig, ax = plt.subplots(1, 1, figsize=(1, 1), frameon=False)
        ax.axis("off")  # Remove annoying black frame.

        mesh = np.zeros((dpi, dpi, 3))

        # Loop over colors
        for icol in range(3):

            data = load_and_extract(fn)
            pids = data.gas.particle_ids
            data.gas.masses[:] = rgbs[pids, icol]

            mesh[:, :, icol] = project_gas_pixel_grid(data, dpi)

            mesh[:, :, icol] = normalize(mesh[:, :, icol])

        # Global variable for set_array
        plot = ax.imshow(
            mesh,
            extent=[0, 1, 0, 1],
            interpolation="none",
        )

        # Remove all whitespace
        fig.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=None, hspace=None)

        fig.savefig("{}_{:04d}.png".format(outfilename, n), dpi=dpi)
        plt.close(fig)

        # Loop over colors
        for icol in range(3):

            # Creation of first frame
            fig, ax = plt.subplots(1, 1, figsize=(1, 1), frameon=False)
            ax.axis("off")  # Remove annoying black frame.

            mesh = np.zeros((dpi, dpi, 3))

            data = load_and_extract(fn)
            pids = data.gas.particle_ids
            data.gas.masses[:] = rgbs[pids, icol]

            mesh[:, :, icol] = project_gas_pixel_grid(data, dpi)

            mesh[:, :, icol] = normalize(mesh[:, :, icol])

            # Global variable for set_array
            plot = ax.imshow(
                mesh,
                extent=[0, 1, 0, 1],
                interpolation="none",
            )

            # Remove all whitespace
            fig.subplots_adjust(left=0, bottom=0, right=1,
                                top=1, wspace=None, hspace=None)

            fig.savefig("{}_{:04d}_{:02d}.png".format(
                outfilename, n, icol), dpi=dpi)
            plt.close(fig)
