import numpy as np
from pydmd import DMD
from rDMD.rDMDClass import rDMDClass
from Utils import _col_major_2darray, plot_mode_2D_flow, calculate_psnr
import csv
from datetime import datetime
from matplotlib import pyplot as plt
import scipy.integrate

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
    iterations = 10
    dmdCSV = open('results_prediction_noise/dmd.csv', 'w')
    rdmdCSV = open('results_prediction_noise/rdmd.csv', 'w')
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
            start = datetime.now()
            dmd = DMD(svd_rank=r, tlsq_rank=0, exact=True, opt=True)
            dmd.fit(train)
            dmd_pred = getPredicted(train, len(X[0]) - training_size, dmd)
            end = datetime.now()
            time_spent = end - start
            time_spent_row.append(time_spent)
            error_row.append(calculateMeanPSNR(dmd_pred))
        row_dmd = row_dmd+time_spent_row+error_row
        dmdWriter.writerow(row_dmd)


        for p_val in p:
            for q_val in q:
                row_rdmd = [r, p_val, q_val]
                time_spent_row = []
                err_spent_row = []
                for i in range(iterations):
                    rdmd = rDMDClass(svd_rank=r, tlsq_rank=0, exact=True, opt=False)
                    start = datetime.now()
                    rdmd.fit(train, oversample=p_val, n_subspace=q_val, random_state=None)
                    rdmd_pred = getPredicted(train, len(X[0]) - training_size, rdmd)
                    end = datetime.now()
                    time_spent = end - start
                    time_spent_row.append(time_spent)
                    err_spent_row.append(calculateMeanPSNR(rdmd_pred))
                row_rdmd = row_rdmd +time_spent_row + err_spent_row
                rdmdWriter.writerow(row_rdmd)
            print("P: " + str(p_val))
        print("RANK: "+str(r))

    row_dmd = ["none"]
    time_spent_row = []
    error_row = []

    for i in range(iterations):
        # get optimal prediction
        start = datetime.now()
        dmd = DMD(tlsq_rank=0, exact=True, opt=True)
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

def reshaping(X):
    return X.reshape((ny, nx), order='F')

def plotIntegral():
    plt.rcParams['font.size'] = '13'
    compute_integral = scipy.integrate.trapz
    dmd_states = [state.reshape((ny,nx)) for state in dmd_pred]
    dmd_int = [compute_integral(compute_integral(state)).real for state in dmd_states]
    rdmd_states_10 = [state.reshape((ny,nx)) for state in rdmd_pred_10]
    rdmd_states_20 = [state.reshape((ny, nx)) for state in rdmd_pred_20]
    rdmd_states_30 = [state.reshape((ny, nx)) for state in rdmd_pred_30]
    rdmd_states_40 = [state.reshape((ny, nx)) for state in rdmd_pred_40]
    original_int = [compute_integral(compute_integral(snapshot)).real for snapshot in (reshaping(x) for x in test.T)]
    rdmd_int_10 = [compute_integral(compute_integral(state)).real for state in rdmd_states_10]
    rdmd_int_20 = [compute_integral(compute_integral(state)).real for state in rdmd_states_20]
    rdmd_int_30 = [compute_integral(compute_integral(state)).real for state in rdmd_states_30]
    rdmd_int_40 = [compute_integral(compute_integral(state)).real for state in rdmd_states_40]
    figure = plt.figure(figsize=(18, 5))
    steps = list(range(122,len(X[0]),1))
    plt.plot(steps, original_int, 'g.', label='migawki czasowe macierzy danych bez szumu')
    plt.plot(steps, rdmd_int_10, 'r+', label='rDMD rząd docelowy 10,bez szumu')
    plt.plot(steps, rdmd_int_20, 'b+', label='rDMD rząd docelowy 20,bez szumu')
    plt.plot(steps, rdmd_int_30, 'k+', label='rDMD rząd docelowy 30,bez szumu')
    plt.plot(steps, rdmd_int_40, 'y+', label='rDMD rząd docelowy 40,bez szumu')
    plt.ylabel('Całka')
    plt.xlabel('Czas')
    plt.grid()
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    leg = plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

    print("MSE")
    print("rank 10")
    print(np.square(np.subtract(original_int, rdmd_int_10)).mean())
    print("rank 20")
    print(np.square(np.subtract(original_int, rdmd_int_20)).mean())
    print("rank 30")
    print(np.square(np.subtract(original_int, rdmd_int_30)).mean())
    print("rank 40")
    print(np.square(np.subtract(original_int, rdmd_int_40)).mean())

def plotPSNR():
    # plot and calculate PSNR
    data = [x for x in (snap.reshape((ny,nx)) for snap in test.T)]
    PSNR_metric_RDMD = []
    PSNR_metric_DMD = []
    PSNR_metric_RDMD_noise = []
    PSNR_metric_DMD_noise = []
    f, ax = plt.subplots(1)
    for id_subplot, snapshot in enumerate(rDMD_pred):
        if id_subplot-1 <151:
            PSNR_metric_RDMD.append(calculate_psnr(data[id_subplot].real, snapshot.reshape((ny,nx)).real))
    for id_subplot, snapshot in  enumerate(DMD_pred):
        if id_subplot-1 < 151:
            PSNR_metric_DMD.append(calculate_psnr(data[id_subplot].real, snapshot.reshape((ny,nx)).real))
    for id_subplot, snapshot in  enumerate(DMD_pred_noise):
        if id_subplot-1 < 151:
            PSNR_metric_DMD_noise.append(calculate_psnr(data[id_subplot].real, snapshot.reshape((ny,nx)).real))
    for id_subplot, snapshot in  enumerate(rDMD_pred_noise):
        if id_subplot-1 < 151:
            PSNR_metric_RDMD_noise.append(calculate_psnr(data[id_subplot].real, snapshot.reshape((ny,nx)).real))

    print(np.mean(PSNR_metric_DMD))
    print(np.mean(PSNR_metric_RDMD))
    print(np.mean(PSNR_metric_RDMD_noise))
    print(np.mean(PSNR_metric_DMD_noise))
    plt.plot(np.linspace(122, 151,29), PSNR_metric_DMD, 'k', label="DMD bez szumu")
    plt.plot(np.linspace(122, 151,29), PSNR_metric_RDMD, 'g', label="rDMD bez szumu")
    plt.plot(np.linspace(122, 151,29), PSNR_metric_DMD_noise, 'r',label="DMD z szumem")
    plt.plot(np.linspace(122,151,29), PSNR_metric_RDMD_noise, 'b', label="rDMD z szumem")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title("Rząd docelowy = 20 \n Parametr nadprókowania = 0 \n Liczba iteracji potęgowych = 1  ")
    ax.set_ylim(bottom=0)
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

X = np.genfromtxt('../../data/vorticity.csv', delimiter=',', dtype=None)
#split _train&test
training_size =122
train = X[:,:122]
test = X[:,122:]


target_snr_db= 10
x_watts =  train **2
x_db = 10 * np.log10(x_watts)
sig_avg_watts = np.mean(x_watts)
sig_avg_db = 10 * np.log10(sig_avg_watts)
noise_avg_db = sig_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
sig_avg = np.mean(train)
mean_noise = 0
print(np.sqrt(noise_avg_watts))
noise =np.random.normal(mean_noise, 5, train.shape)
train_noise = train + noise

#collect data for plotting
# get_data_for_dmd()

# #learn dmd
# dmd = DMD( tlsq_rank=0, exact=True, opt=True)
# dmd.fit(train)
# #learn rdmd
# rdmd_10 = rDMDClass(svd_rank=10, tlsq_rank=0, exact=True, opt=False)
# rdmd_10.fit(train, oversample = 0,n_subspace=1, random_state =None)
#
# rdmd_20 = rDMDClass(svd_rank=20, tlsq_rank=0, exact=True, opt=False)
# rdmd_20.fit(train, oversample = 0,n_subspace=1, random_state =None)
#
# rdmd_30 = rDMDClass(svd_rank=30, tlsq_rank=0, exact=True, opt=False)
# rdmd_30.fit(train, oversample = 0,n_subspace=1, random_state =None)
#
# rdmd_40 = rDMDClass(svd_rank=40, tlsq_rank=0, exact=True, opt=False)
# rdmd_40.fit(train, oversample = 0,n_subspace=1, random_state =None)
#
# #get predictions
# dmd_pred = getPredicted(train, len(X[0])-training_size, dmd)
# rdmd_pred_10 = getPredicted(train, len(X[0])-training_size, rdmd_10)
# rdmd_pred_20 = getPredicted(train, len(X[0])-training_size, rdmd_20)
# rdmd_pred_30 = getPredicted(train, len(X[0])-training_size, rdmd_30)
# rdmd_pred_40 = getPredicted(train, len(X[0])-training_size, rdmd_40)
#get error
# dmd_psnr =calculatePSNR(dmd_pred)
# rdmd_psnr =calculatePSNR(rdmd_pred)

# print(len(dmd_pred))
# print(len(dmd_pred[0]))
#
# nx = 449
# ny = 199
# dmd_pred = np.asarray(dmd_pred)
# rdmd_pred_10 = np.asarray(rdmd_pred_10)
# rdmd_pred_20 = np.asarray(rdmd_pred_20)
# rdmd_pred_30 = np.asarray(rdmd_pred_30)
# rdmd_pred_40 = np.asarray(rdmd_pred_40)
# plotIntegral()

# #learn dmd
dmd = DMD( svd_rank=20,tlsq_rank=0, exact=True, opt=True)
dmd.fit(train)
#learn rdmd
rdmd = rDMDClass(svd_rank=20, tlsq_rank=0, exact=True, opt=False)
rdmd.fit(train, oversample = 0,n_subspace=1, random_state =None)

dmd_noise = DMD( svd_rank=20,tlsq_rank=0, exact=True, opt=True)
dmd_noise.fit(train_noise)

rdmd_noise = rDMDClass(svd_rank=20, tlsq_rank=0, exact=True, opt=False)
rdmd_noise.fit(train_noise, oversample = 0,n_subspace=1, random_state =None)

#get predictions
DMD_pred = getPredicted(train, len(X[0])-training_size, dmd)
rDMD_pred = getPredicted(train, len(X[0])-training_size, rdmd)
DMD_pred_noise = getPredicted(train_noise, len(X[0])-training_size, dmd_noise)
rDMD_pred_noise = getPredicted(train_noise, len(X[0])-training_size, rdmd_noise)
#get error
nx = 449
ny = 199
plotPSNR()


