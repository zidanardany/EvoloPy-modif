# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris
"""

import numpy as np
import math
from scipy.io import loadmat
from basic_functions import fsphere, fgriewank, frastrigin, fackley, fweierstrass


# define the function blocks
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def F1(x):
    s = np.sum(x ** 2)
    return s


def F2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o


def F3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (np.sum(x[0:i])) ** 2
    return o


def F4(x):
    o = max(abs(x))
    return o


def F5(x):
    dim = len(x)
    o = np.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def F6(x):
    o = np.sum(abs((x + 0.5)) ** 2)
    return o


def F7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = np.sum(w * (x ** 4)) + np.random.uniform(0, 1)
    return o


def F8(x):
    o = sum(-x * (np.sin(np.sqrt(abs(x)))))
    return o


def F9(x):
    dim = len(x)
    o = np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    return o


def F10(x):
    dim = len(x)
    o = (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
        - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
        + 20
        + np.exp(1)
    )
    return o


def F11(x):
    dim = len(x)
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
    return o


def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + np.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((np.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + np.sum(Ufun(x, 10, 100, 4))
    return o


def F13(x):
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (np.sin(3 * np.pi * x[:,0])) ** 2
        + np.sum(
            (x[:,:-1] - 1) ** 2
            * (1 + (np.sin(3 * np.pi * x[:,1:])) ** 2), axis=1
        )
        + ((x[:,-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[:,-1])) ** 2)
    ) + np.sum(Ufun(x, 5, 100, 4))
    return o


def F14(x):
    aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    aS = np.asarray(aS)
    bS = np.zeros(25)
    v = np.matrix(x)
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = np.sum((np.power(H, 6)))
    w = [i for i in range(25)]
    for i in range(0, 24):
        w[i] = i + 1
    o = ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)
    return o


def F15(L):
    aK = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    aK = np.asarray(aK)
    bK = np.asarray(bK)
    bK = 1 / bK
    fit = np.sum(
        (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
    )
    return fit


def F16(L):
    o = (
        4 * (L[0] ** 2)
        - 2.1 * (L[0] ** 4)
        + (L[0] ** 6) / 3
        + L[0] * L[1]
        - 4 * (L[1] ** 2)
        + 4 * (L[1] ** 4)
    )
    return o


def F17(L):
    o = (
        (L[1] - (L[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * L[0] - 6)
        ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[0])
        + 10
    )
    return o


def F18(L):
    o = (
        1
        + (L[0] + L[1] + 1) ** 2
        * (
            19
            - 14 * L[0]
            + 3 * (L[0] ** 2)
            - 14 * L[1]
            + 6 * L[0] * L[1]
            + 3 * L[1] ** 2
        )
    ) * (
        30
        + (2 * L[0] - 3 * L[1]) ** 2
        * (
            18
            - 32 * L[0]
            + 12 * (L[0] ** 2)
            + 48 * L[1]
            - 36 * L[0] * L[1]
            + 27 * (L[1] ** 2)
        )
    )
    return o


# map the inputs to the function blocks
def F19(L):
    aH = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    pH = [
        [0.3689, 0.117, 0.2673],
        [0.4699, 0.4387, 0.747],
        [0.1091, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
    pH = np.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F20(L):
    aH = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    pH = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    pH = np.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F21(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(5):
        v = np.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F22(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(7):
        v = np.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F23(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(10):
        v = np.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o

#-----------------------------------------------------------------------------------------------
# define the composition function
def hybrid_composition_func(x, fun_num, func, o, sigma, lamda, bias, M):
    D = len(x)
    weight = np.zeros(fun_num)
    for i in range(fun_num):
        oo = o[i, :]
        weight[i] = np.exp(-np.sum(np.power(x - oo, 2)) / (2 * D * sigma[i] ** 2))

    weight = np.where(weight == np.max(weight), 
                      weight, 
                      weight * (1 - np.max(weight) ** 10))

    weight /= np.sum(weight)

    fit = 0
    for i in range(fun_num):
        oo = o[i, :]
        f = func[f'f{i+1}'](((x - oo) / (lamda[i])) @ M[f'M{i+1}'])
        x1 = 5 * np.ones(D)
        f1 = func[f'f{i+1}']((x1 / (lamda[i]))  @ M[f'M{i+1}'])
        fit1 = 2000 * f / f1
        fit += weight[i] * (fit1 + bias[i])

    return fit
    
#-----------------------------------------------------------------------------------------------
#   1.	Composition Function 1
def CF1(x):
    D = len(x)
    fun_num = 10
    mat_dict = loadmat('com_func1_data.mat')  # load predefined optima

    o = mat_dict['o']
    if len(o[0,:]) >= D:
        o = o[:, :D]
    else:
        o = -5 + 10 * np.random.rand(fun_num, D)
    o[9, :] = np.zeros(D)

    func = {'f1': fsphere,
            'f2': fsphere,
            'f3': fsphere,
            'f4': fsphere,
            'f5': fsphere,
            'f6': fsphere,
            'f7': fsphere,
            'f8': fsphere,
            'f9': fsphere,
            'f10': fsphere
           }

    bias = np.arange(fun_num) * 100
    sigma = np.ones(fun_num)
    lamda = np.ones(fun_num) * 5/100

    M = {}
    for i in range(1, fun_num+1):
        M[f'M{i}'] = np.diag(np.ones(D))

    fit = hybrid_composition_func(x, fun_num, func, o, sigma, lamda, bias, M)
    return fit

#-----------------------------------------------------------------------------------------------
#   2.	Composition Function 2
def CF2(x):
    D = len(x)
    fun_num = 10
    mat_dict = loadmat('com_func2_data.mat')  # load predefined optima

    o = mat_dict['o']
    if len(o[0,:]) >= D:
        o = o[:, :D]
    else:
        o = -5 + 10 * np.random.rand(fun_num, D)
    o[9, :] = np.zeros(D)

    func = {'f1': fgriewank,
            'f2': fgriewank,
            'f3': fgriewank,
            'f4': fgriewank,
            'f5': fgriewank,
            'f6': fgriewank,
            'f7': fgriewank,
            'f8': fgriewank,
            'f9': fgriewank,
            'f10': fgriewank
           }

    bias = np.arange(fun_num) * 100
    sigma = np.ones(fun_num)
    lamda = np.ones(fun_num) * 5/100

    if D == 10:
        mat_dict = loadmat('com_func2_M_D10.mat')
        M = {}
        for i in range(fun_num):
            M[f'M{i+1}'] = list(mat_dict['M'].tolist()[0][0])[i]
    else:
        M = {}
        for i in range(1, fun_num+1):
            M[f'M{i}'] = orthm_generator(D)

    fit = hybrid_composition_func(x, fun_num, func, o, sigma, lamda, bias, M)
    return fit
    
#-----------------------------------------------------------------------------------------------
#   3.	Composition Function 3
def CF3(x):
    D = len(x)
    fun_num = 10
    mat_dict = loadmat('com_func3_data.mat')  # load predefined optima

    o = mat_dict['o']
    if len(o[0,:]) >= D:
        o = o[:, :D]
    else:
        o = -5 + 10 * np.random.rand(fun_num, D)
    o[9, :] = np.zeros(D)

    func = {'f1': frastrigin,
            'f2': frastrigin,
            'f3': frastrigin,
            'f4': frastrigin,
            'f5': frastrigin,
            'f6': frastrigin,
            'f7': frastrigin,
            'f8': frastrigin,
            'f9': frastrigin,
            'f10': frastrigin
           }

    bias = np.arange(fun_num) * 100
    sigma = np.ones(fun_num)
    lamda = np.array([5/32, 5/32, 1, 1, 10, 10, 5/100, 5/100, 5/100, 5/100])

    if D == 10:
        mat_dict = loadmat('com_func3_M_D10.mat')
        M = {}
        for i in range(fun_num):
            M[f'M{i+1}'] = list(mat_dict['M'].tolist()[0][0])[i]
    else:
        M = {}
        for i in range(1, fun_num+1):
            M[f'M{i}'] = orthm_generator(D)

    fit = hybrid_composition_func(x, fun_num, func, o, sigma, lamda, bias, M)
    return fit

#-----------------------------------------------------------------------------------------------
#   4.	Rotated Hybrid Composition Function 1
def CF4(x):
    D = len(x)
    fun_num = 10
    mat_dict = loadmat('hybrid_func1_data.mat')  # load predefined optima

    o = mat_dict['o']
    if len(o[0,:]) >= D:
        o = o[:, :D]
    else:
        o = -5 + 10 * np.random.rand(fun_num, D)
    o[9, :] = np.zeros(D)

    func = {'f1': fackley,
            'f2': fackley,
            'f3': frastrigin,
            'f4': frastrigin,
            'f5': fweierstrass,
            'f6': fweierstrass,
            'f7': fgriewank,
            'f8': fgriewank,
            'f9': fsphere,
            'f10': fsphere
           }

    bias = np.arange(fun_num) * 100
    sigma = np.ones(fun_num)
    lamda = np.array([5/32, 5/32, 1, 1, 10, 10, 5/100, 5/100, 5/100, 5/100])

    if D == 10:
        mat_dict = loadmat('hybrid_func1_M_D10.mat')
        M = {}
        for i in range(fun_num):
            M[f'M{i+1}'] = list(mat_dict['M'].tolist()[0][0])[i]
    else:
        M = {}
        for i in range(1, fun_num+1):
            M[f'M{i}'] = orthm_generator(D)

    fit = hybrid_composition_func(x, fun_num, func, o, sigma, lamda, bias, M)
    return fit

#-----------------------------------------------------------------------------------------------
#   5.	Rotated Hybrid Composition Function 2
def CF5(x):
    D = len(x)
    fun_num = 10
    mat_dict = loadmat('hybrid_func2_data.mat')  # load predefined optima

    o = mat_dict['o']
    if len(o[0,:]) >= D:
        o = o[:, :D]
    else:
        o = -5 + 10 * np.random.rand(fun_num, D)
    o[9, :] = np.zeros(D)

    func = {'f1': frastrigin,
            'f2': frastrigin,
            'f3': fweierstrass,
            'f4': fweierstrass,
            'f5': fgriewank,
            'f6': fgriewank,
            'f7': fackley,
            'f8': fackley,
            'f9': fsphere,
            'f10': fsphere
           }

    bias = np.arange(fun_num) * 100
    sigma = np.ones(fun_num)
    lamda = np.array([1/5, 1/5, 10, 10, 5/100, 5/100, 5/32, 5/32, 5/100, 5/100])

    if D == 10:
        mat_dict = loadmat('hybrid_func2_M_D10.mat')
        M = {}
        for i in range(fun_num):
            M[f'M{i+1}'] = list(mat_dict['M'].tolist()[0][0])[i]
    else:
        M = {}
        for i in range(1, fun_num+1):
            M[f'M{i}'] = orthm_generator(D)

    fit = hybrid_composition_func(x, fun_num, func, o, sigma, lamda, bias, M)
    return fit

#-----------------------------------------------------------------------------------------------
#   6.	Rotated Hybrid Composition Function 3
def CF6(x):
    D = len(x)
    fun_num = 10
    mat_dict = loadmat('hybrid_func2_data.mat')  # load predefined optima

    o = mat_dict['o']
    if len(o[0,:]) >= D:
        o = o[:, :D] 
    else:
        o = -5 + 10 * np.random.rand(fun_num, D)
    o[9, :] = np.zeros(D)

    func = {'f1': frastrigin,
            'f2': frastrigin,
            'f3': fweierstrass,
            'f4': fweierstrass,
            'f5': fgriewank,
            'f6': fgriewank,
            'f7': fackley,
            'f8': fackley,
            'f9': fsphere,
            'f10': fsphere
           }

    bias = np.arange(fun_num) * 100
    sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    lamda = np.array([1/5, 1/5, 10, 10, 5/100, 5/100, 5/32, 5/32, 5/100, 5/100]) * sigma

    if D == 10:
        mat_dict = loadmat('hybrid_func2_M_D10.mat')
        M = {}
        for i in range(fun_num):
            M[f'M{i+1}'] = list(mat_dict['M'].tolist()[0][0])[i]
    else:
        M = {}
        for i in range(1, fun_num+1):
            M[f'M{i}'] = orthm_generator(D)

    fit = hybrid_composition_func(x, fun_num, func, o, sigma, lamda, bias, M)
    return fit
    
def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
        "CF1": ["CF1", -5, 5, 10],
        "CF2": ["CF2", -5, 5, 10],
        "CF3": ["CF3", -5, 5, 10],
        "CF4": ["CF4", -5, 5, 10],
        "CF5": ["CF5", -5, 5, 10],
        "CF6": ["CF6", -5, 5, 10],
    }
    return param.get(a, "nothing")

'''
# ESAs space mission design benchmarks https://www.esa.int/gsp/ACT/projects/gtop/
from fcmaes.astro import (
    MessFull,
    Messenger,
    Gtoc1,
    Cassini1,
    Cassini2,
    Rosetta,
    Tandem,
    Sagas,
)
def Ca1(x):
    return Cassini1().fun(x)
def Ca2(x):
    return Cassini2().fun(x)
def Ros(x):
    return Rosetta().fun(x)
def Tan(x):
    return Tandem(5).fun(x)
def Sag(x):
    return Sagas().fun(x)
def Mef(x):
    return MessFull().fun(x)
def Mes(x):
    return Messenger().fun(x)
def Gt1(x):
    return Gtoc1().fun(x)

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
        "Ca1": [
            "Ca1",
            Cassini1().bounds.lb,
            Cassini1().bounds.ub,
            len(Cassini1().bounds.lb),
        ],
        "Ca2": [
            "Ca2",
            Cassini2().bounds.lb,
            Cassini2().bounds.ub,
            len(Cassini2().bounds.lb),
        ],
        "Gt1": ["Gt1", Gtoc1().bounds.lb, Gtoc1().bounds.ub, len(Gtoc1().bounds.lb)],
        "Mes": [
            "Mes",
            Messenger().bounds.lb,
            Messenger().bounds.ub,
            len(Messenger().bounds.lb),
        ],
        "Mef": [
            "Mef",
            MessFull().bounds.lb,
            MessFull().bounds.ub,
            len(MessFull().bounds.lb),
        ],
        "Sag": ["Sag", Sagas().bounds.lb, Sagas().bounds.ub, len(Sagas().bounds.lb)],
        "Tan": [
            "Tan",
            Tandem(5).bounds.lb,
            Tandem(5).bounds.ub,
            len(Tandem(5).bounds.lb),
        ],
        "Ros": [
            "Ros",
            Rosetta().bounds.lb,
            Rosetta().bounds.ub,
            len(Rosetta().bounds.lb),
        ],
    }
    return param.get(a, "nothing")
'''
