import matplotlib.pyplot as plt
import h5py
import numpy as np

def plot_slice(arr: np.ndarray, axes: plt.Axes) -> plt.Axes:
    """Plot data, with x in first dimension and y in second dimension"""
    axes.imshow(arr.T, origin="lower")
    return axes

def plot_save_all_y_slices(data):
    fig, ax = plt.subplots(1, 1)
    # Make the y axis the first axis
    for i, y_slice in enumerate(np.moveaxis(data, 1, 0)):
        plot_slice(y_slice, ax)
        fig.savefig(f"plot-out/out{i}.png")
        ax.clear()

def calc_intensity_squared(x, y, z):
    return x * x + y * y + z * z

def dataset_take(dataset: h5py.Dataset, indices, axis=0):
    """Like numpy.take() but for h5py datasets.
    
    Useful so that the entire dataset doesn't have to be loaded into memory as a numpy array to do slicing"""
    if isinstance(indices, int):
        # Turn number to tuple with that number
        indices = (indices,)
    else:
        indices = tuple(indices)

    # see https://stackoverflow.com/questions/11249446/python-extracting-one-slice-of-a-multidimensional-array-given-the-dimension-ind
    slicing = (slice(None),) * axis + indices + (slice(None),)
    return dataset[slicing]

def plot_intensity_yslices(ex: h5py.Dataset, ey: h5py.Dataset, ez: h5py.Dataset):
    fig, ax = plt.subplots(1, 1)
    # Second dimension is y_index. Assume all datasets have same dimensions
    for y_index in range(ex.shape[1]):
        intensities = calc_intensity_squared(ex[:, y_index, :], ey[:, y_index, :], ez[:, y_index, :])
        plot_slice(intensities, ax)
        fig.savefig(f"plot-out/out{y_index}.png")
        ax.clear()

ex_file = h5py.File("run_beam-out/run_beam-ex-000020.00.h5", "r")
ex_r = ex_file["ex.r"]
ex_i = ex_file["ex.i"]

ey_file = h5py.File("run_beam-out/run_beam-ey-000020.00.h5", "r")
ey_r = ey_file["ey.r"]
ey_i = ey_file["ey.i"]

ez_file = h5py.File("run_beam-out/run_beam-ez-000020.00.h5", "r")
ez_r = ez_file["ez.r"]
ez_i = ez_file["ez.i"]

plot_intensity_yslices(ex_r, ey_r, ez_r)