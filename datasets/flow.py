from pydmd import DMD
from rDMD.rDMDClass import rDMDClass
from Utils import _col_major_2darray, plot_mode_2D_flow
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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
    #plt.rcParams['font.size'] = '30'
    plt.yscale('log')
    plt.plot(s, 'o')
    plt.show()

    var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=3)
    print(sum(var_explained[:10]))
    plt.bar(x=list(range(1, len(var_explained[:40]) + 1)),
                height=var_explained[:40], color="limegreen")
    plt.show()

def plotDMDMode(m):
    m= reshaping(m.real.copy())
    plt.imshow(m)
    plt.show()

def plotAverageDMDmode(modes, scales):
    avgMode = np.array([np.mean(modes,axis =1 )]).T
    plot_mode_2D_flow(x=np.arange(nx), y=np.arange(ny),
                      figsize=(20, 5), modes=avgMode, index_mode=None, scales = scales)

def plotEigs(eigs):
    plt.plot(np.real(eigs), np.imag(eigs), '.', markersize=13)
    plt.xlabel("część rzeczywista")
    plt.ylabel("część urojona")
    theta = np.linspace(0, 2 * np.pi, 1024)
    plt.plot(np.cos(theta), np.sin(theta), "k--")
    plt.show()

X = np.genfromtxt('../../data/vorticity.csv', delimiter=',', dtype=None)
print(len(X))
print(len(X[0]))
#plotSnapshot(X[:,150].copy())
#plotSVD(X)

#wyliczenie DMD modes czyli POD
dmd = DMD(svd_rank=10, tlsq_rank=0, exact=True, opt=True)
dmd.fit(X)
# plotDMDMode(dmd.modes[:,1])
#plot_mode_2D_flow(x=np.arange(nx), y=np.arange(ny),
#                  figsize=(20,5), modes=dmd.modes,index_mode =None)

rdmd = rDMDClass(svd_rank=10, tlsq_rank=0, exact=True, opt=False)
rdmd.fit(X, oversample = 12,n_subspace=2, random_state =None)
#plot_mode_2D_flow(x=np.arange(nx), y=np.arange(ny),
#                  figsize=(20,5), modes=rdmd.modes,index_mode =None)
#print average mode for DMD and rDMD
#scales = [[min(dmd.modes.real.min(),rdmd.modes.real.min())/10,max(dmd.modes.real.max(),rdmd.modes.real.max())/10]
#    ,[min(dmd.modes.imag.min(),rdmd.modes.imag.min()),max(dmd.modes.imag.max(),rdmd.modes.imag.max())]]
#plotAverageDMDmode(dmd.modes,scales)
#plotAverageDMDmode(rdmd.modes,scales)

#plot eigs
plotEigs(dmd.eigs)

