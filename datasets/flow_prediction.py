import numpy as np
from pydmd import DMD
from rDMD.rDMDClass import rDMDClass
from Utils import _col_major_2darray, plot_mode_2D_flow, calculate_psnr

def getPredicted(frameStart, numOfPredictions, decomposition):
    predicted = []
    for i in range(numOfPredictions):
        pred = decomposition.predict(frameStart)
        predicted.append(pred[:,training_size-1])
        frameStart = pred
    return predicted

def calculatePSNR(pred):
    PSNR = []
    for i in range(len(test[0])):
        PSNR.append(calculate_psnr(test[:,i], pred[i].real))
    return PSNR

X = np.genfromtxt('../../data/vorticity.csv', delimiter=',', dtype=None)
#split _train&test
training_size =122
train = X[:,:122]
test = X[:,122:]
#learn dmd
dmd = DMD(svd_rank=20, tlsq_rank=0, exact=True, opt=True)
dmd.fit(train)
#learn rdmd
rdmd = rDMDClass(svd_rank=20, tlsq_rank=0, exact=True, opt=False)
rdmd.fit(train, oversample = 10,n_subspace=0, random_state =None)
#get predictions
dmd_pred = getPredicted(train, len(X[0])-training_size, dmd)
rdmd_pred = getPredicted(train, len(X[0])-training_size, dmd)
#get error
dmd_psnr =calculatePSNR(dmd_pred)
rdmd_psnr =calculatePSNR(dmd_pred)