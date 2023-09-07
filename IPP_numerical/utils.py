import numpy as np
import numpy.random as npr


def make_D_rho(D, rho):
    return np.delete(np.abs(D - D[rho,:]), rho, axis = 0)

def rand_01_mat(m,n):
    return npr.randint(2,size = (m,n))

def rand_01_col_mat(m,n):
    prob = npr.rand(n)
    return npr.binomial(1,prob, size = (m,n))
