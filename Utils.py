import numpy as np
import matplotlib.pyplot as plt
from past.utils import old_div
from scipy import linalg
from os.path import splitext
import math
from sklearn.metrics import mean_squared_error

def normalize0to255(img):
    image = img * (255.0/img.max())
    return image

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1=normalize0to255(img1)
    img2=normalize0to255(img2)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_mse(img1, img2):
    # img1 and img2 have range [0, 255]
    #img1=normalize0to255(img1)
    #img2=normalize0to255(img2)
    mse = mean_squared_error(img1,img2)#np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return mse #-10 * math.log10(mse/(255.0*255.0) )

def _col_major_2darray(X):
    """
    Private method that takes as input the snapshots and stores them into a
    2D matrix, by column. If the input data is already formatted as 2D
    array, the method saves it, otherwise it also saves the original
    snapshots shape and reshapes the snapshots.
    :param X: the input snapshots.
    :type X: int or numpy.ndarray
    :return: the 2D matrix that contains the flatten snapshots, the shape
        of original snapshots.
    :rtype: numpy.ndarray, tuple
    """
    # If the data is already 2D ndarray
    if isinstance(X, np.ndarray) and X.ndim == 2:
        snapshots = X
        snapshots_shape = None
    else:
        input_shapes = [np.asarray(x).shape for x in X]

        if len(set(input_shapes)) != 1:
            raise ValueError("Snapshots have not the same dimension.")

        snapshots_shape = input_shapes[0]
        snapshots = np.transpose([np.asarray(x).flatten() for x in X])

    # check condition number of the data passed in
    cond_number = np.linalg.cond(snapshots)
    if cond_number > 10e4:
        print(
            "Input data matrix X has condition number {}. "
            """Consider preprocessing data, passing in augmented data
matrix, or regularization methods.""".format(
                cond_number
            )
        )

    return snapshots, snapshots_shape

def plot_mode_2D(
    index_mode = None,
    filename = None,
    x = None,
    y = None,
    order = "C",
    figsize = (8, 8),
    modes = None,
    scales=None
):
    """
Plot the DMD Modes.
"""
    if modes is None:
        raise ValueError(
            "The modes have not been computed."
            "You have to perform the fit method."
        )
    xgrid, ygrid = np.meshgrid(x, y)

    if index_mode is None:
        index_mode = list(range(modes.shape[1]))
    elif isinstance(index_mode, int):
        index_mode = [index_mode]

    if filename:
        basename, ext = splitext(filename)

    for idx in index_mode:
        fig = plt.figure(figsize=figsize)
        fig.suptitle("DMD wektor własny {}".format(idx))

        real_ax = fig.add_subplot(1, 2, 1)
        imag_ax = fig.add_subplot(1, 2, 2)

        mode = modes.T[idx].reshape(xgrid.shape, order=order)

        real = real_ax.pcolor(
            xgrid,
            ygrid,
            mode.real,
            cmap="jet",
            vmin=scales[0][0],#mode.real.min(),
            vmax=scales[0][1],#mode.real.max(),
        )
        imag = imag_ax.pcolor(
            xgrid,
            ygrid,
            mode.imag,
            vmin=scales[1][0],#mode.imag.min(),
            vmax=scales[1][1],#mode.imag.max(),
        )

        fig.colorbar(real, ax=real_ax)
        fig.colorbar(imag, ax=imag_ax)

        real_ax.set_aspect("auto")
        imag_ax.set_aspect("auto")

        real_ax.set_title("Część rzeczywista")
        imag_ax.set_title("Część urojona")

        # padding between elements
        plt.tight_layout(pad=2.0)

        if filename:
            plt.savefig("{0}.{1}{2}".format(basename, idx, ext))
            plt.close(fig)

    if not filename:
        plt.show()

def plot_mode_2D_flow(
    index_mode = None,
    filename = None,
    x = None,
    y = None,
    order = "F",
    figsize = (8, 8),
    modes = None,
    scales=None
):
    """
Plot the DMD Modes.
"""
    plt.rcParams['font.size'] = '30'
    if modes is None:
        raise ValueError(
            "The modes have not been computed."
            "You have to perform the fit method."
        )
    xgrid, ygrid = np.meshgrid(x, y)

    if index_mode is None:
        index_mode = list(range(modes.shape[1]))
    elif isinstance(index_mode, int):
        index_mode = [index_mode]

    if filename:
        basename, ext = splitext(filename)
    for idx in index_mode:
        fig = plt.figure(figsize=figsize)
        fig.suptitle("rDMD wektor własny {}".format(idx))

        real_ax = fig.add_subplot(1, 2, 1)
        imag_ax = fig.add_subplot(1, 2, 2)

        mode =modes[:,idx].reshape((199,449), order=order) #modes.T[idx].reshape((x,y), order=order)

        real = real_ax.pcolor(
            xgrid,
            ygrid,
            mode.real,
            cmap="jet",
            vmin=mode.real.min() if scales == None else scales[0][0],#mode.real.min(),
            vmax=mode.real.max() if scales == None else scales[0][1],#mode.real.max(),
        )
        imag = imag_ax.pcolor(
            xgrid,
            ygrid,
            mode.imag,
            vmin=mode.imag.min() if scales == None else scales[1][0],#mode.imag.min(),
            vmax=mode.imag.max() if scales == None else scales[1][1],#mode.imag.max(),
        )

        fig.colorbar(real, ax=real_ax)
        fig.colorbar(imag, ax=imag_ax)

        real_ax.set_aspect("auto")
        imag_ax.set_aspect("auto")

        real_ax.set_title("Część rzeczywista")
        imag_ax.set_title("Część urojona")

        # padding between elements
        plt.tight_layout()

        if filename:
            plt.savefig("{0}.{1}{2}".format(basename, idx, ext))
            plt.close(fig)

    if not filename:
        plt.show()