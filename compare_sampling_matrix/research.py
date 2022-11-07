from rsvd import rsvd
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import rc

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

def firstTest():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)
    m = 200
    n = 200
    A = generate_decaying_matrix(m, n)

    mins = []
    errs = []

    ls = range(1, 151, 5)
    for l in ls:
        _, _, _, Q = rsvd(A, l, n_oversamples=0, return_range=True, samplingMatrixType="Uniform")
        err = np.linalg.norm((np.eye(m) - Q @ Q.T) @ A, 2)
        errs.append(np.log10(err))

        S = np.linalg.svd(A, compute_uv=False)
        min_ = S[l + 1]
        mins.append(np.log10(min_))

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

firstTest()