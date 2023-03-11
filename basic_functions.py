import math
import numpy as np
  
def w(x, c1, c2):
    return np.sum(c1 * np.cos(c2 * x))
  
def fweierstrass(x):
    dim = len(x)
    x = x + 0.5
    a = 0.5
    b = 3
    kmax = 20
    c1 = a ** np.arange(kmax+1)
    c2 = 2 * np.pi * b ** np.arange(kmax+1)
    f = 0
    c = -w(0.5, c1, c2)
    for i in range(dim):
        f = f + w(x[i], c1, c2)
    f = f + c * dim
    return f
  
def frastrigin(x):
    f = np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
    return f
  
def fackley(x):
    dim = len(x)
    f = np.sum(x**2)
    f = 20 - 20 * np.exp(-0.2 * np.sqrt(f/dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + np.exp(1)
    return f
  
def fgriewank(x):
    dim = len(x)
    f = 1
    for i in range(dim):
        f = f * np.cos(x[i] / np.sqrt(i + 1))
    f = np.sum(x**2) / 4000 - f + 1
    return f
  
def fsphere(x):
    f = np.sum(x**2)
    return f

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "fsphere": "fsphere",
        "fgriewank": "fgriewank",
        "frastrigin": "frastrigin",
        "fackley": "fackley",
        "fweierstrass": "fweierstrass"
    }
    return param.get(a, "nothing")
