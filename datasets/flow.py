from pydmd import DMD
from datetime import datetime
from rDMD.rDMDClass import rDMDClass
from PIL import Image
from matplotlib import pyplot as plt
from Utils import _col_major_2darray, plot_mode_2D_flow, calculate_psnr
import numpy as np
import scipy.integrate
import time
import matplotlib.ticker as mtick
import csv

nx = 449
ny = 199

xx, yy = np.meshgrid(np.linspace(-1, 8, nx), np.linspace(-2, 2, ny))
cylinder_idx = (xx**2+yy**2)<.5**2

def reshaping(X):
    return X.reshape((ny, nx), order='F')
# X = np.concatenate((np.load("../data/mat_UALL_1.npy"), np.load("../data/mat_UALL_2.npy")))

def plotSnapshot(x):

    X_transformed = reshaping(x.copy())
    #to remove cylinder uncommend
    #X_transformed[cylinder_idx] = 50
    norm_for_plotting = (255 * (X_transformed - np.min(X_transformed)) / np.ptp(X_transformed)).astype(int)
    img = Image.fromarray(np.uint8(norm_for_plotting))
    img.show()

def plotSVD(snapshots):
    fig = plt.figure(figsize=(18, 12))
    u, s, v = np.linalg.svd(snapshots.T, full_matrices=False)
    plt.rcParams['font.size'] = '30'
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.yscale('log')
    plt.plot(s, 'o')
    plt.ylabel("\u03C3"+"_k")
    plt.xlabel("k")
    plt.show()

    var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=3)
    print(sum(var_explained[:10]))
    plt.bar(x=list(range(1, len(var_explained[:40]) + 1)),
                height=var_explained[:40], color="limegreen")
    plt.xlabel("Numer wartości własnej")
    plt.ylabel("Procent wyjaśnialności zbioru")
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.show()

def plotDMDMode(m):
    m= reshaping(m.real.copy())
    plt.imshow(m)
    plt.show()

def plotAverageDMDmode(modes, scales):
    avgMode = np.array([np.mean(modes,axis =1 )]).T
    plot_mode_2D_flow(x=np.arange(nx), y=np.arange(ny),
                      figsize=(25, 10), modes=avgMode, index_mode=None, scales = scales)

def plotEigs(eigs1, eigs2):
    plt.rcParams['font.size'] = '13'
    plt.plot(np.real(eigs1), np.imag(eigs1), 'rx', markersize=13, label="DMD")
    plt.plot(np.real(eigs2), np.imag(eigs2), 'g.', markersize=13, label ="rDMD")
    plt.xlabel("część rzeczywista")
    plt.ylabel("część urojona")
    theta = np.linspace(0, 2 * np.pi, 1024)
    plt.plot(np.cos(theta), np.sin(theta), "k--")
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

def plotIntegral():
    compute_integral = scipy.integrate.trapz
    dmd_states = [state.reshape((ny,nx)) for state in dmd.reconstructed_data.T]
    dmd_int = [compute_integral(compute_integral(state)).real for state in dmd_states]
    rdmd_states = [state.reshape((ny,nx)) for state in rdmd.reconstructed_data.T]
    original_int = [compute_integral(compute_integral(snapshot)).real for snapshot in (reshaping(x) for x in X.T)]
    rdmd_int = [compute_integral(compute_integral(state)).real for state in rdmd_states]
    figure = plt.figure(figsize=(18, 5))
    plt.plot(rdmd.original_timesteps, original_int, 'g.', label='migawki czasowe macierzy danych')
    plt.plot(rdmd.dmd_timesteps, rdmd_int, 'r+', label='wektory własne uzyskane przez rDMD')
    plt.plot(dmd.dmd_timesteps, dmd_int, 'b+', label='wektory własne uzyskane przez DMD')
    plt.ylabel('Całka')
    plt.xlabel('Czas')
    plt.grid()
    leg = plt.legend()
    plt.show()

def plotPSNR():
    # plot and calculate PSNR
    data = [x for x in (reshaping(snap) for snap in X.T)]
    PSNR_metric_RDMD = []
    PSNR_metric_DMD = []
    for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
        PSNR_metric_DMD.append(calculate_psnr(data[id_subplot - 1].real, snapshot.reshape((ny,nx)).real))
    for id_subplot, snapshot in enumerate(rdmd.reconstructed_data.T, start=1):
        PSNR_metric_RDMD.append(calculate_psnr(data[id_subplot - 1].real, snapshot.reshape((ny,nx)).real))
    plt.plot(np.linspace(1, 151, 151), np.subtract(PSNR_metric_DMD,PSNR_metric_RDMD), 'r')
    leg = plt.legend()
    plt.show()

def showAverageExecutionTime(repetitions):
    dmd_time, rdmd_time = 0, 0
    for i in range(repetitions):
        start = time.time()
        dmd.fit(X)
        end = time.time()
        dmd_time += (end - start)
    for i in range(repetitions):
        start = time.time()
        rdmd.fit(X, oversample=12, n_subspace=2, random_state=None)
        end = time.time()
        rdmd_time += (end - start)
    print("Sredni czas wykonania dmd: " + str(dmd_time / repetitions) + " sekund")
    print("Sredni czas wykonania rdmd: " + str(rdmd_time / repetitions) + " sekund")

def get_error(oryg, approx):
    licznik = np.linalg.norm(oryg-approx, 2)
    mianownik  =  np.linalg.norm(oryg, 2)
    return licznik/mianownik

def get_data_for_dmd():
    target_rank = [10, 20, 30, 40]
    q = [0, 1, 2, 3]
    p = [0, 2, 5, 8, 10]
    iterations = 10
    dmdCSV = open('results_reconstruction/dmd.csv', 'w')
    rdmdCSV = open('results_reconstruction/rdmd.csv', 'w')
    dmdWriter = csv.writer(dmdCSV)
    rdmdWriter = csv.writer(rdmdCSV)
    time_headers = [("time_"+str(i)) for i in range(iterations)]
    error_headers = [("error_" + str(i)) for i in range(iterations)]

    dmdWriter.writerow(["target_rank"]+time_headers)
    rdmdWriter.writerow(["target_rank", "p_val","q_val"] + time_headers+error_headers)

    for r in target_rank:
        row_dmd = [r]
        time_spent_row = []
        for i in range(iterations):
            start = datetime.now()
            dmd = DMD(svd_rank=r, tlsq_rank=0, exact=True, opt=True)
            dmd.fit(X)
            end = datetime.now()
            time_spent = end - start
            time_spent_row.append(time_spent)
        row_dmd = row_dmd+time_spent_row
        dmdWriter.writerow(row_dmd)

        for p_val in p:
            for q_val in q:
                row_rdmd = [r, p_val, q_val]
                time_spent_row = []
                err_spent_row = []
                for i in range(iterations):
                    rdmd = rDMDClass(svd_rank=r, tlsq_rank=0, exact=True, opt=False)
                    start = datetime.now()
                    rdmd.fit(X, oversample=p_val, n_subspace=q_val, random_state=None)
                    end = datetime.now()
                    time_spent = end - start
                    time_spent_row.append(time_spent)
                    err = get_error(rdmd.reconstructed_data, dmd.reconstructed_data)
                    err_spent_row.append(err)
                row_rdmd = row_rdmd +time_spent_row + err_spent_row
                rdmdWriter.writerow(row_rdmd)

    dmdCSV.close()
    rdmdCSV.close()


X = np.genfromtxt('../../data/vorticity.csv', delimiter=',', dtype=None)
print(len(X))
print(len(X[0]))
#plotSnapshot(X[:,150].copy())
#plotSVD(X)

#wyliczenie DMD modes czyli POD
dmd = DMD(svd_rank=10, tlsq_rank=0, exact=True, opt=True)
dmd.fit(X)
#plotDMDMode(dmd.modes[:,1])
#plot_mode_2D_flow(x=np.arange(nx), y=np.arange(ny), filename="dmd.jpg",
#                  figsize=(20,8), modes=dmd.modes,index_mode =None)


rdmd = rDMDClass(svd_rank=10, tlsq_rank=0, exact=True, opt=False)
rdmd.fit(X, oversample = 10,n_subspace=2, random_state =None)
# plot_mode_2D_flow(x=np.arange(nx), y=np.arange(ny), filename="dmd.jpg",
#                   figsize=(20,8), modes=rdmd.modes,index_mode =None)

#print average mode for DMD and rDMD
# scales = [[min(dmd.modes.real.min(),rdmd.modes.real.min())/10,max(dmd.modes.real.max(),rdmd.modes.real.max())/10]
#     ,[min(dmd.modes.imag.min(),rdmd.modes.imag.min()),max(dmd.modes.imag.max(),rdmd.modes.imag.max())]]
# plotAverageDMDmode(dmd.modes,scales)
# plotAverageDMDmode(rdmd.modes,scales)

get_data_for_dmd()

#plot eigs
#plotEigs(dmd.eigs)
#plotEigs(dmd.eigs,rdmd.eigs)

#reconstructed data integral
#plotIntegral()

#reconstructed data MSE by PSNR
#plotPSNR()

# #compare time spent on task
# repetitions =5
# showAverageExecutionTime(repetitions)