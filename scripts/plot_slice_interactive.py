import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py

parser = argparse.ArgumentParser(
    prog="h5plot",
    description="""plot y (2nd) slice of 3d h5 files.
    Colormap and normalization is set from initial slice of data and doesn't update when using the GUI slider
    """
)

parser.add_argument("filename")
parser.add_argument("dataset_name")
parser.add_argument("yslice", type=int, help="y index in dataset of slice to plot")
parser.add_argument("-m", "--cmap", dest="cmap", default="viridis")

args = parser.parse_args()

f = h5py.File(args.filename, "r")

dataset = f[args.dataset_name]
y_slice = dataset[:, args.yslice, :]

fig, ax = plt.subplots()

ax.set_title(args.dataset_name)
ax.set_xlabel("x (dataset index)")
ax.set_ylabel("z (dataset index)")

img = ax.imshow(
    # Since imshow expects row column order, transpose so that x is actually columns and z is rows
    y_slice.T,
    cmap=args.cmap,
    origin="lower",
    interpolation="none",
    )
fig.colorbar(img)

# Make room for slider left of plot
fig.subplots_adjust(left=0.25)
ax_y = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
y_slider = Slider(
    ax=ax_y,
    label="y coordinate",
    valmin=0,
    valmax=dataset.shape[1] - 1,
    valstep=1,
    valinit=args.yslice,
    closedmax=True,
    orientation="vertical"
)

def update(y: float):
    y_slice = dataset[:, round(y), :]
    # Set data does not update the normalization
    img.set_data(y_slice.T)
    # fig.colorbar(img)
    fig.canvas.draw_idle()

y_slider.on_changed(update)

plt.show()