import numpy as np
from pydmd import DMD
from rDMD.rDMDClass import rDMDClass
from Utils import _col_major_2darray, plot_mode_2D_flow, calculate_psnr
import csv
from datetime import datetime

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

def calculateMeanPSNR(pred):
    p=calculatePSNR(pred)
    return getMeanPSNR(p)

def getMeanPSNR(p):
    return np.mean(p)

def get_data_for_dmd():
    target_rank = [10, 20, 30, 40]
    q =[0, 1, 2, 3]
    p =[0, 2, 5, 8, 10]
    iterations = 5#10
    dmdCSV = open('results_prediction/dmd.csv', 'w')
    rdmdCSV = open('results_prediction/rdmd.csv', 'w')
    dmdWriter = csv.writer(dmdCSV)
    rdmdWriter = csv.writer(rdmdCSV)
    time_headers = [("time_"+str(i)) for i in range(iterations)]
    error_headers = [("error_" + str(i)) for i in range(iterations)]

    dmdWriter.writerow(["target_rank"]+time_headers+error_headers)
    rdmdWriter.writerow(["target_rank", "p_val","q_val"] + time_headers+error_headers)

    for r in target_rank:
        row_dmd = [r]
        time_spent_row = []
        error_row = []
        for i in range(iterations):
            dmd = DMD(svd_rank=r, tlsq_rank=0, exact=True, opt=True)
            start = datetime.now()
            dmd.fit(train)
            dmd_pred = getPredicted(train, len(X[0]) - training_size, dmd)
            end = datetime.now()
            time_spent = end - start
            time_spent_row.append(time_spent)
            error_row.append(calculateMeanPSNR(dmd_pred))
        row_dmd = row_dmd+time_spent_row+error_row
        dmdWriter.writerow(row_dmd)


        # for p_val in p:
        #     for q_val in q:
        #         row_rdmd = [r, p_val, q_val]
        #         time_spent_row = []
        #         err_spent_row = []
        #         for i in range(iterations):
        #             rdmd = rDMDClass(svd_rank=r, tlsq_rank=0, exact=True, opt=False)
        #             start = datetime.now()
        #             rdmd.fit(X, oversample=p_val, n_subspace=q_val, random_state=None)
        #             rdmd_pred = getPredicted(train, len(X[0]) - training_size, rdmd)
        #             end = datetime.now()
        #             time_spent = end - start
        #             time_spent_row.append(time_spent)
        #             err_spent_row.append(calculateMeanPSNR(rdmd_pred))
        #         row_rdmd = row_rdmd +time_spent_row + err_spent_row
        #         rdmdWriter.writerow(row_rdmd)

    row_dmd = ["none"]
    time_spent_row = []
    error_row = []

    for i in range(iterations):
        # get optimal prediction
        dmd = DMD(tlsq_rank=0, exact=True, opt=True)
        start = datetime.now()
        dmd.fit(train)
        dmd_pred = getPredicted(train, len(X[0]) - training_size, dmd)
        end = datetime.now()
        time_spent = end - start
        time_spent_row.append(time_spent)
        error_row.append(calculateMeanPSNR(dmd_pred))

    row_dmd = row_dmd + time_spent_row + error_row
    dmdWriter.writerow(row_dmd)

    dmdCSV.close()
    rdmdCSV.close()


X = np.genfromtxt('../../data/vorticity.csv', delimiter=',', dtype=None)
#split _train&test
training_size =122
train = X[:,:122]
test = X[:,122:]
#collect data for plotting
get_data_for_dmd()

# #learn dmd
# dmd = DMD( tlsq_rank=0, exact=True, opt=True)
# dmd.fit(train)
# #learn rdmd
# rdmd = rDMDClass(svd_rank=20, tlsq_rank=0, exact=True, opt=False)
# rdmd.fit(train, oversample = 10,n_subspace=0, random_state =None)
# #get predictions
# dmd_pred = getPredicted(train, len(X[0])-training_size, dmd)
# rdmd_pred = getPredicted(train, len(X[0])-training_size, dmd)
# #get error
# dmd_psnr =calculatePSNR(dmd_pred)
# rdmd_psnr =calculatePSNR(dmd_pred)