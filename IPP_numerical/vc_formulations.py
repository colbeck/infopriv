import numpy as np
import numpy.random as npr
import cvxpy as cp

def Adjacency_to_D(A, w):
    return 0
    # A is adjacency matrix

def vertex_cover(A, w):
    # A is adjacency matrix 
    n = A.shape[0]
    x = cp.Variable(n, boolean = True)
    obj = cp.Minimize(w @ x)
    cons = []
    for i in range(n):
        for j in range(i+1):
            if A[i,j] == 1:
                cons += [x[i]+x[j] >= 1]

    prob = cp.Problem(obj, cons)
    prob.solve()
    return prob.value, x.value

def vertex_cover_lp(A, w):
    # K is adjacency matrix 
    n = A.shape[0]
    x = cp.Variable(n)
    obj = cp.Minimize(w @ x)
    cons = [x >= 0, x <= 1]
    for i in range(n):
        for j in range(i+1):
            if A[i,j] == 1:
                cons += [x[i]+x[j] >= 1]

    prob = cp.Problem(obj, cons)
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
        for j in range(i):
            if A[i,j] == 1:
                cons += [X[n,i] + X[n,j] >= 1]
                cons += [1 - X[n,i] - X[n,j] + X[i,j] == 0]
    
    prob = cp.Problem(obj,cons)
    prob.solve()
    return prob.value, X.value[n,:-1]

def theta(A):
    n, _  = A.shape
    X = cp.Variable((n,n), PSD = True)
    J = np.ones((n,n))

    obj = cp.Maximize(cp.trace(J @ X))
    cons = [cp.trace(X) == 1]
    for i in range(n):
        for j in range(i):
            if A[i,j] == 1:
                cons += [X[i,j] == 0]

    prob = cp.Problem(obj,cons)
    prob.solve()
    return prob.value, np.diag(X.value)

    


