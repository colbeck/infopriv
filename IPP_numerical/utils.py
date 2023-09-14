import numpy as np
import numpy.random as npr
import cvxpy as cp


def make_D_rho(D, rho):
    return np.delete(np.abs(D - D[rho,:]), rho, axis = 0)

def rand_01_mat(m,n, prob = 0.5):
    return rand_01_col_mat(m,n, prob)
    # return npr.randint(2,size = (m,n))

def rand_01_col_mat(m,n, prob = None):
    if not prob:
        prob = npr.rand(n)
    return npr.binomial(1,prob, size = (m,n))


def Adjacency_to_D(A, w):
    return 0
    # A is adjacency matrix

def vertex_cover(A, w):
    m,n = A.shape
    assert(m == n)
    x = cp.Variable(n, boolean=True)

    obj = cp.Minimize(w @ x)
    cons = []
    for i in range(n):
        for j in range(n):
            if A[i,j] == 1:
                cons += [x[i] + x[j] >= 1]
    
    prob = cp.Problem(obj,cons)
    prob.solve()
    return prob.value, x.value

def vertex_cover_sdp(A, w):
    m,n = A.shape
    assert(m == n)
    X = cp.Variable((n + 1, n + 1), PSD = True)

    obj = cp.Minimize(w @ X[n,:-1])
    cons = []
    for i in range(n):
        X[i,i] == X[n,i]
        for j in range(n):
            if A[i,j] == 1:
                cons += [X[n,i] + X[n,j] >= 1]
    
    prob = cp.Problem(obj,cons)
    prob.solve()
    return prob.value, X.value[n,:-1]


def theta(A):
    return 0
    


