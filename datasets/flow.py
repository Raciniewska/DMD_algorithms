import numpy as np
from PIL import Image

nx = 449
ny = 199


def reshaping(X):
    return X.reshape((ny, nx), order='F')

X = np.concatenate((np.load("../data/mat_UALL_1.npy"), np.load("../data/mat_UALL_2.npy")))
print(len(X))
print(len(X[0]))
X_transformed = reshaping(X[:,100].copy())
norm_for_plotting =(255*(X_transformed  - np.min(X_transformed ))/np.ptp(X_transformed )).astype(int)
img = Image.fromarray(np.uint8(norm_for_plotting))
img.show()
