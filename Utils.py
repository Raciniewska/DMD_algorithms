import numpy as np
import matplotlib.pyplot as plt
from past.utils import old_div
from scipy import linalg


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
