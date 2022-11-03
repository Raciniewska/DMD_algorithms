from scipy import linalg
import numpy as np
from sklearn.utils import check_random_state

def orthonormalize(A, overwrite_a=True, check_finite=False):
    #Ortogonalizacja kolumn macierzy A przez dekompozycje QR
    Q, _ = linalg.qr(A, overwrite_a=overwrite_a, check_finite=check_finite,
                     mode='economic', pivoting=False)
    return

def conjugate_transpose(A):
    #Sprzężona dekompozycja macierzy A
    if A.dtype == np.complexfloating:
        return A.conj().T
    return A.T

def random_gaussian_map(A, l, axis, random_state):
    #generowanie macierzy Gaussa
    return random_state.standard_normal(size=(A.shape[axis], l))\
        .astype(A.dtype)

def perform_subspace_iterations(A, Q, n_iter=2, axis=1):
    #Przeprowadzenie teracji potegowych na macierzy Q
    if axis == 0:
        Q = Q.T
    Q = orthonormalize(Q)
    for _ in range(n_iter):
        if axis == 0:
            Z = orthonormalize(A.dot(Q))
            Q = orthonormalize(A.T.dot(Z))
        else:
            Z = orthonormalize(A.T.dot(Q))
            Q = orthonormalize(A.dot(Z))
    if axis == 0:
        return Q.T
    return

def johnson_lindenstrauss(A, l, axis=1, random_state=None):
    """
        Dla macierzy A o wymiarach m x n i liczby l, wyliczana jest ortagonalna macierz Q o wymiarach  m x l, której rząd przybliża rząd macierzy A
    """
    random_state = check_random_state(random_state)
    A = np.asarray(A)
    # construct gaussian random matrix
    Omega = random_gaussian_map(A, l, axis, random_state)
    #Projekcja macierzy A na macierz Q
    if axis == 0:
        return Omega.T.dot(A)
    return A.dot(Omega)


def _compute_rqb(A, rank, oversample, n_subspace,  random_state):
    Q = johnson_lindenstrauss(A, rank + oversample, random_state=random_state)
    if n_subspace > 0:
        Q = perform_subspace_iterations(A, Q, n_iter=n_subspace, axis=1)
    else:
        Q = orthonormalize(Q)
    # Projekcja macierzy danych na podprzestrzen niskowymiarową
    B = conjugate_transpose(Q).dot(A)
    return Q, B


def compute_rqb(A, rank, oversample=20, n_subspace=2, random_state=None):
    Q, B = _compute_rqb(np.asarray_chkfinite(A),
                    rank=rank, oversample=oversample,
                        n_subspace=n_subspace, random_state=random_state)
    return Q, B