import numpy as np


def kpf_graph(n, k=1):
    # generate matrices for k-predecessor following topology graph
    assert k <= n, "number of predecessor neighbors must be less than size of graph"
    A = np.zeros((n, n))
    D = np.zeros((n, n))
    L = np.zeros((n, n))
    P = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i-1, max(-1, i-k-1), -1):
            A[i, j] = 1
    D = np.diag(A.sum(axis=1))
    L = D - A
    for i in range(k):
        P[i, i] = 1
    return A, D, L, P