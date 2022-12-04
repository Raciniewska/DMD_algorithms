from rsvd import rsvd
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import rc
from sklearn.preprocessing import normalize
from PIL import Image
import time
import csv
from statistics import variance


def generate_decaying_matrix(m, n):
    U = np.eye(m)
    # Rapidly decaying singular values with some noise.
    S = [-0.75 ** i * np.random.randn() for i in range(n)]
    Vt = np.eye(n)
    return U @ np.diag(S) @ Vt

def gen_linear_decaying_spectrum_matrix(m, n, k):
    U = np.eye(m)
    # Rapidly decaying singular values with some noise.
    S = [i if i <= k else 0 for i in range(n)]
    Vt = np.eye(n)
    return U @ np.diag(S) @ Vt

def getTimeData():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)
    A = np.asarray(Image.open('../data/Albert-Einstein.jpg').convert('L'))
    print(A)
    ls = [10,20,40,60,80,100,200,300,400,500]
    result_dict = {
        "Uniform":[],
        "Gauss":[],
        "SRFT":[],
        "SRHT":[]
    }
    for l in ls:
        names = ["Uniform","Gauss", "SRFT","SRHT"]
        for name in names:
            time_spent=[]
            for i in range(5):
                start = time.time()
                _, _, _, Q = rsvd(A, l, n_oversamples=0, return_range=True, n_subspace_iters=2,samplingMatrixType=name)
                end = time.time()
                time_spent.append(end -start)
            result_dict[name].append(np.mean(time_spent))
            print(name+" : l="+str(l))
        name = "Deterministic"
        time_spent =[]
        for i in range(5):
            start = time.time()
            S = np.linalg.svd(A, compute_uv=False)
            end = time.time()
            time_spent.append(end - start)
        print(np.mean(time_spent))
        print(name+" : l="+str(l))
    with open("resultsTime.csv", "w") as outfile:
        writer = csv.writer(outfile)
        key_list = list(result_dict.keys())
        limit = len(result_dict[key_list[0]])
        writer.writerow(result_dict.keys())
        for i in range(limit):
            row =[]
            for key in key_list:
                row.append(result_dict[key][i])
            writer.writerow(row)

def getErrorData():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)
    m = 512#256
    n =512 #256
    A = np.asarray(Image.open('../data/Albert-Einstein.jpg').convert('L'))
    print(A)
    mins = []

    ls = [10,20,40,60,80,100,200,300,400,500]
    result_dict = {
        "Uniform": [],
        "Gauss": [],
        "SRFT": [],
        "SRHT": [],
        "Deterministic" :[]
    }
    S = np.linalg.svd(A, compute_uv=False)
    for l in ls:
        names = ["Uniform", "Gauss", "SRFT", "SRHT"]
        for name in names:
            errs = []
            for i in range(5):
                _, _, _, Q = rsvd(A, l, n_oversamples=0, return_range=True, n_subspace_iters=2, samplingMatrixType=name)
                err = np.linalg.norm((np.eye(m) - Q @ Q.T) @ A, 2)
                errs.append(np.log10(err))
            result_dict[name].append(np.mean(errs))
            print(name + " : l=" + str(l))
        name = "Deterministic"
        min_ = S[l + 1]
        result_dict[name].append(np.log10(min_))
        print(name + " : l=" + str(l))
    with open("resultsError.csv", "w") as outfile:
        writer = csv.writer(outfile)
        key_list = list(result_dict.keys())
        limit = len(result_dict[key_list[0]])
        writer.writerow(result_dict.keys())
        for i in range(limit):
            row =[]
            for key in key_list:
                row.append(result_dict[key][i])
            writer.writerow(row)

def getErrorDataWithVariance():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)
    m = 2048#512#256
    n =2048#512 #256
    A = np.asarray(Image.open('../data/flowers.jpg').convert('L'))
    print(A)
    mins = []

    ls = [10,20,40,60,80,100,200,300,400,500]
    result_dict = {
        "Uniform": [],
        "Gauss": [],
        "SRFT": [],
        "SRHT": []
    }
    S = np.linalg.svd(A, compute_uv=False)
    for l in ls:
        names = ["Uniform", "Gauss", "SRFT", "SRHT"]
        for name in names:
            errs = []
            for i in range(5):
                _, _, _, Q = rsvd(A, l, n_oversamples=0, return_range=True, n_subspace_iters=2, samplingMatrixType=name)
                err = np.linalg.norm((np.eye(m) - Q @ Q.T) @ A, 2)
                errs.append(np.log10(err))
            result_dict[name].append(variance(errs))
            print(name + " : l=" + str(l))
    with open("resultsErrorVar.csv", "w") as outfile:
        writer = csv.writer(outfile)
        key_list = list(result_dict.keys())
        limit = len(result_dict[key_list[0]])
        writer.writerow(result_dict.keys())
        for i in range(limit):
            row =[]
            for key in key_list:
                row.append(result_dict[key][i])
            writer.writerow(row)

def firstTest():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)
    m = 2048#512#256
    n = 2048#512 #256
    A = np.asarray(Image.open('../data/flowers.jpg').convert('L'))
    print(A)
    mins = []
    errs = []

    ls = [10,20,40,60,80,100,200,300,400,500]
    for l in ls:
        _, _, _, Q = rsvd(A, l, n_oversamples=0, return_range=True, n_subspace_iters=2,samplingMatrixType="Uniform")

        err = np.linalg.norm((np.eye(m) - Q @ Q.T) @ A, 2)
        errs.append(np.log10(err))
        S = np.linalg.svd(A, compute_uv=False)
        min_ = S[l + 1]
        mins.append(np.log10(min_))

        print(l)

    ax.scatter(ls, mins, color='#11accd', s=30,
               label=r'$\log_{10}(\sigma_{\ell+1})$', marker='v')
    ax.scatter(ls, errs, color='#807504', s=30, label=r'$\log_{10}(e_{\ell})$',
               marker='o')
    ax.plot(ls, mins, color='#11accd', linewidth=1)
    ax.plot(ls, errs, color='#807504', linewidth=1)

    ax.set_ylabel('Order of magnitude of errors')
    ax.set_xlabel(r'Random samples $\ell$')
    ax.set_title('Exponentially decaying singular values')

    plt.legend()
    plt.tight_layout()
    plt.show()

def showPowerIterations():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)

    k = 30
    A =gen_linear_decaying_spectrum_matrix(50, 50, k)

    S_true = np.linalg.svd(A, compute_uv=False)
    S_true = [s / S_true.max() for s in S_true]
    x = range(len(S_true))
    ax.scatter(x, S_true, color='gray')
    ax.plot(x, S_true, label='True singular values', color='gray', marker='s')

    qs = [(1, '#11accd', '*'), (2, '#807504', 'v'), (3, '#bc2612', 'd')]
    for q, color, marker in qs:
        A_new = (A @ A.T) ** q @ A
        S = np.linalg.svd(A_new, compute_uv=False)
        S_new = [s / S.max() for s in S]
        ax.scatter(x, S_new, color=color)
        ax.plot(x, S_new, label=r'$q = %s$' % q, color=color, marker=marker)

    ax.set_ylabel('Normalized magnitude')
    ax.set_xlabel(r'Singular values $\sigma_i$')
    ax.set_title('Normalized singular values per $q$ power iterations')
    plt.legend()
    plt.tight_layout()
    plt.show()

getErrorDataWithVariance()