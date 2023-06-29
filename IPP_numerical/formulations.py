import numpy as np
import numpy.random as npr
import cvxpy as cp
import time
import matplotlib.pyplot as plt

# Matrix Generation
def rand_01_mat(m,n):
    return npr.randint(0,2,size = (m,n))

def rand_01_col_mat(m,n):
    prob = npr.rand(n)
    return npr.binomial(1,prob, size = (m,n))

# MISP
def misp(D_rho, w, r):
    m,n = D_rho.shape
    x = cp.Variable(n, boolean = True)
    y = cp.Variable(m, boolean = True)
    obj = cp.Minimize(w @ x)
    cons = [D_rho @ x >= y, cp.sum(y) >= m-r]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return prob.value, x.value, y.value

def misp_lp(D_rho, w, r):
    m,n = D_rho.shape
    x = cp.Variable(n)
    y = cp.Variable(m)
    obj = cp.Minimize(w @ x)
    cons = [D_rho @ x >= y, cp.sum(y) >= m-r]
    cons += [x >= 0, x <= 1, y >= 0, y <= 1]
    for j in range(m):
        for i in range(n):
            if np.allclose(D_rho[j,i], 1):
                cons += [y[j] >= x[i]]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return prob.value, x.value, y.value

def misp_sdp(D_rho, w, r):
    m,n = D_rho.shape

    X = cp.Variable((n+m+1,n+m+1), PSD = True)

    obj = cp.Minimize(w @ X[n+m,:n])
    cons = []
    cons += [cp.sum(X[n+m,n:n+m]) >= m-r]

    #KEY
    # x = X[n+m,:n] or X[:n,n+m]
    # y = X[n+m,n:n+m] or X[n:n+m,n+m] 
    # X = X[:n,:n]
    # Y = X[n:n+m,n:n+m]
    # Z = X[:n,n:n+m], Z.T = X[n:n+m, :n]
    m1 = np.ones(m)
    n1 = np.ones(n)
    m1_s = np.ones((m,1))
    n1_s = np.ones((n,1))

    for i in range(n):
        cons += [X[i,i] == X[n+m,i]]
    for j in range(n, n+m):
        cons += [X[j,j] == X[n+m,j]]
    cons += [X[m+n,m+n] == 1]
    for i in range(n):
        for j in range(m):
            if np.allclose(D_rho[j,i], 1):
                cons += [X[i,n+j] == X[n+m,i]]
                cons += [X[n+m,n+j] >= X[n+m,i]]
    cons += [X >= 0, X <= 1]
    cons += [D_rho @ X[n+m,:n] >= X[n+m,n:n+m]]
    # cons += [D_rho @ X[:n,n:n+m] @ m1 >= (m-r) * X[n+m,n:n+m]] # (1,2)
    # cons += [(D_rho @ X[:n,[n+m]]) @ m1_s.T >= X[n:n+m,n:n+m]] # (2,3)
    # cons += [(D_rho @ X[:n,[n+m]]) @ m1_s.T >= X[:n,n:n+m].T # (2,3)
    # cons += [m1 @ X[n:n+m,n+m] * m1 >= X[n:n+m,n+m] * (m-r)] # (1,3)
    # cons += [m1 @ X[n:n+m,n+m] * n1 >= X[:n,n+m] * (m-r)] # (1,3)
    prob = cp.Problem(obj, cons)

    prob.solve()
    return prob.value, X.value


def masp(D_rho, w, r):
    m,n = D_rho.shape
    x = cp.Variable(n, boolean = True)
    y = cp.Variable(m, boolean = True)
    obj = cp.Maximize(w @ x)
    cons = [D_rho @ x <= y*n, cp.sum(y) <= m-r]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return prob.value, x.value, y.value

def masp_lp(D_rho, w, r):
    m,n = D_rho.shape
    x = cp.Variable(n)
    y = cp.Variable(m)
    obj = cp.Mazimize(w @ x)
    cons = [D_rho @ x <= y, cp.sum(y) <= m-r]
    cons += [x >= 0, x <= 1, y >= 0, y <= 1]
    for j in range(m):
        for i in range(n):
            if np.allclose(D_rho[j,i], 1):
                cons += [y[j] >= x[i]]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return prob.value, x.value, y.value

def masp_sdp(D_rho, w, r):
    m,n = D_rho.shape

    X = cp.Variable((n+m+1,n+m+1), PSD = True)

    obj = cp.Maximize(w @ X[n+m,:n])
    cons = []
    cons += [cp.sum(X[n+m,n:n+m]) <= m-r]

    for i in range(n):
        cons += [X[i,i] == X[n+m,i]]
    for j in range(n, n+m):
        cons += [X[j,j] == X[n+m,j]]
    cons += [X[m+n,m+n] == 1]
    for i in range(n):
        for j in range(m):
            if np.allclose(D_rho[j,i], 1):
                cons += [X[i,n+j] == X[n+m,i]]
                cons += [X[n+m,n+j] >= X[n+m,i]]
    cons += [X >= 0, X <= 1]
    cons += [D_rho @ X[n+m,:n] <= X[n+m,n:n+m]]


    prob = cp.Problem(obj, cons)

    prob.solve()
    return prob.value, X.value

