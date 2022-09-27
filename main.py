import numpy as np #1
import scipy.integrate
from matplotlib import pyplot as plt
from pydmd import DMD
from rDMD.rDMDClass import rDMDClass
from Utils import _col_major_2darray
import time as tm

#Generowanie szeregów czasowych + nakładanie szumu
x1 = np.linspace(-3, 3, 80)
x2 = np.linspace(-3, 3, 80)
x1grid, x2grid = np.meshgrid(x1, x2)
time = np.linspace(0, 6, 16)
data = [2/np.cosh(x1grid)/np.cosh(x2grid)*(1.2j**-t) for t in time]
noise = [np.random.normal(0.0, 0.4, size=x1grid.shape) for t in time]
snapshots = [d+n for d,n in zip(data, noise)]

#Wizualizacja wejściowych szeregów czasowych
fig = plt.figure(figsize=(18,12))
for id_subplot, snapshot in enumerate(snapshots, start=1):
    plt.subplot(4, 4, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot.real, vmin=-1, vmax=1)
plt.suptitle("Input snapshots",fontsize=40)
plt.show()
# The `svd_rank` can be set to zero for an automatic selection of the truncation rank;
# in some cases (as this tutorial) the singular values should be examinated in order to select the proper truncation.
fig = plt.figure(figsize=(18,12))
plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in snapshots]).T), 'o')
plt.title("Singular values of flatten input snapshots",fontsize=40)
plt.show()

#Deterministyczny DMD na szeregu czasowym
# - `svd_rank`: since the dynamic mode decomposition relies on *singular value decomposition*, we can specify the number of the largest singular values used to approximate the input data.
# - `tlsq_rank`: using the total least square, it is possible to perform a linear regression in order to remove the noise on the data; because this regression is based again on the singular value decomposition, this parameter indicates how many singular values are used.
# - `exact`: boolean flag that allows to choose between the exact modes or the projected one.
# - `opt`: boolean flag that allows to choose between the standard version and the optimized one.
dmd = DMD(svd_rank=1, tlsq_rank=0, exact=True, opt=True)
dmd.fit(snapshots)
dmd.plot_modes_2D(figsize=(12,5))

#Randomizowany DMD na szeregu czasowym
rdmd = rDMDClass(svd_rank=1, tlsq_rank=0, exact=True, opt=False)
rdmd.fit(snapshots, oversample = 12,n_subspace=2, random_state =None)
rdmd.plot_modes_2D(figsize=(12,5))


#MSE w porównaniu z danymi wejściowymi
#DMD deterministyczny
mse=[]
time_spent =[]
for i in range(0,10):
    start = tm.time()
    dmd = DMD(svd_rank=1, tlsq_rank=1, exact=True, opt=True)
    dmd.fit(snapshots)
    end = tm.time()
    time_spent.append(end-start)
for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
    #compute MSE
    difference_array = np.subtract(snapshot.reshape(x1grid.shape).real, snapshots[id_subplot-1].real)
    squared_array = np.square(difference_array)
    mse.append(squared_array.mean())
plt.figure(figsize=(18,12))
plt.plot(mse)
plt.show()

print("Średni MSE dla deterministycznego DMD")
print(sum(mse)/len(mse))

print("Czas [s] deterministycznego DMD")
print(sum(time_spent)/len(time_spent))

#DMD niedeterministyczny

oversample_list = [5, 10, 12]
n_subspace_list =[2,3,4,5]
rmse_list = np.zeros((len(oversample_list),len(n_subspace_list)))
time_list = np.zeros((len(oversample_list),len(n_subspace_list)))

for ovsmpl in range(0, len(oversample_list)):
    for sbspace in range(0,len(n_subspace_list)):
        r_mse=[]
        start = tm.time()
        rdmd = rDMDClass(svd_rank=1, tlsq_rank=0, exact=True, opt=True)
        rdmd.fit(snapshots, oversample = oversample_list[ovsmpl],  n_subspace=n_subspace_list[sbspace], random_state=None)
        end = tm.time()
        time_list[ovsmpl][sbspace] = end-start
        for id_subplot, snapshot in enumerate(rdmd.reconstructed_data.T, start=1):
            # compute MSE
            difference_array = np.subtract(snapshot.reshape(x1grid.shape).real, snapshots[id_subplot - 1].real)
            squared_array = np.square(difference_array)
            r_mse.append(squared_array.mean())

#plt.figure(figsize=(18,12))
#plt.plot(r_mse)
#plt.show()

        print("Średni MSE dla niedeterministycznego DMD")
        print(sum(r_mse)/len(r_mse))
        rmse_list[ovsmpl][sbspace] = sum(r_mse)/len(r_mse)


np.savetxt('out/r_mse.csv', rmse_list, delimiter=',')
np.savetxt('out/r_time.csv', time_list, delimiter=',')

rdmd = rDMDClass(svd_rank=1, tlsq_rank=0, exact=True, opt=True)
rdmd.fit(snapshots, oversample = 12,  n_subspace=2, random_state=None)

#rekonstrukcja stanów deterministyczny DMD
fig = plt.figure(figsize=(18,12))
for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
      plt.subplot(4, 4, id_subplot)
      plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
plt.suptitle("Reconstructed snapshots with deteministic DMD",fontsize=40)
plt.show()

#rekonstrukcja stanów randomizowany DMD
fig = plt.figure(figsize=(18,12))
fig = plt.figure(figsize=(18,12))
for id_subplot, snapshot in enumerate(rdmd.reconstructed_data.T, start=1):
      plt.subplot(4, 4, id_subplot)
      plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
plt.suptitle("Reconstructed snapshots with randomized DMD",fontsize=40)
plt.show()

#Bład rekonstrukcji deterministyczny DMD
# # We can also manipulate the interval between the approximated states and extend the temporal window
# where the data is reconstructed thanks to DMD.
# Let us make the DMD delta time a quarter of the original and extend the temporal window
# to $[0, 3t_{\text{org}}]$, where $t_{\text{org}}$ indicates the time when the last snapshot was caught.
#
print("Shape before manipulation: {}".format(dmd.reconstructed_data.shape))
dmd.dmd_time['dt'] *= 1
dmd.dmd_time['tend'] *= 3
print("Shape after manipulation: {}".format(dmd.reconstructed_data.shape))

fig = plt.figure(figsize=(18,12))
for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
     plt.subplot(7, 7, id_subplot)
     plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
plt.show()

# # For a check about the accuracy of the reconstruction, we can plot the integral computed on the original snapshots and on the DMD states.
compute_integral = scipy.integrate.trapz
dmd_states = [state.reshape(x1grid.shape) for state in dmd.reconstructed_data.T]
original_int = [compute_integral(compute_integral(snapshot)).real for snapshot in snapshots]
dmd_int = [compute_integral(compute_integral(state)).real for state in dmd_states]
figure = plt.figure(figsize=(18, 5))
plt.plot(dmd.original_timesteps, original_int, 'bo', label='original snapshots')
plt.plot(dmd.dmd_timesteps, dmd_int, 'r.', label='dmd states')
plt.ylabel('Integral')
plt.xlabel('Time')
plt.grid()
leg = plt.legend()
plt.show()

#Bład rekonstrukcji randomizowany DMD
print("Shape before manipulation: {}".format(rdmd.reconstructed_data.shape))
rdmd.dmd_time['dt'] *= 1
rdmd.dmd_time['tend'] *= 3
print("Shape after manipulation: {}".format(rdmd.reconstructed_data.shape))

fig = plt.figure(figsize=(18,12))
for id_subplot, snapshot in enumerate(rdmd.reconstructed_data.T, start=1):
     plt.subplot(7, 7, id_subplot)
     plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
plt.show()

# # For a check about the accuracy of the reconstruction, we can plot the integral computed on the original snapshots and on the DMD states.
compute_integral = scipy.integrate.trapz
rdmd_states = [state.reshape(x1grid.shape) for state in rdmd.reconstructed_data.T]
original_int = [compute_integral(compute_integral(snapshot)).real for snapshot in snapshots]
rdmd_int = [compute_integral(compute_integral(state)).real for state in rdmd_states]
figure = plt.figure(figsize=(18, 5))
plt.plot(rdmd.original_timesteps, original_int, 'bo', label='original snapshots')
plt.plot(rdmd.dmd_timesteps, rdmd_int, 'r.', label='rdmd states')
plt.ylabel('Integral')
plt.xlabel('Time')
plt.grid()
leg = plt.legend()
plt.show()

#Predykcja przyszłości przez deterministyczny DMD
fig = plt.figure(figsize=(18,12))
flatten_A, flatten_A_shape= _col_major_2darray(snapshots)
prev_frame = flatten_A[:,15]
predictions = []
for i in range(0,15):
    next_frame = dmd.predict(prev_frame)
    predictions.append(next_frame)
    prev_frame = next_frame
predictions =np.concatenate((flatten_A.T,np.array(predictions)))
for id_subplot, snapshot in enumerate(predictions, start=1):
     plt.subplot(7, 7, id_subplot)
     plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
plt.suptitle("Prediction snapshots with deterministic DMD", fontsize=40)
plt.show()

#Predykcja przyszłości przez niedeterministyczny DMD
plt.figure(2,figsize=(18,12))
prev_frame = flatten_A[:,15]
predictions = []
for i in range(0,15):
    next_frame = rdmd.predict(prev_frame)
    predictions.append(next_frame)
    prev_frame = next_frame
predictions =np.concatenate((flatten_A.T,np.array(predictions)))
for id_subplot, snapshot in enumerate(predictions, start=1):
     plt.subplot(7, 7, id_subplot)
     plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
plt.suptitle("Prediction snapshots with randomized DMD", fontsize=40)
plt.show()
print("finito")
