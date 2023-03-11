# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris
"""

import numpy as np
import math
from scipy.io import loadmat
from basic_functions import fsphere, fgriewank, frastrigin, fackley, fweierstrass


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
    
#   1.	Composition Function 1
def com_func1(x):
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
def com_func2(x):
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
def com_func3(x):
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
def hybrid_func1(x):
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
def hybrid_func2(x):
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
def hybrid_func3(x):
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
        "F1": ["com_func1", -5, 5, 10],
        "F2": ["com_func2", -5, 5, 10],
        "F3": ["com_func3", -5, 5, 10],
        "F4": ["hybrid_func1", -5, 5, 10],
        "F5": ["hybrid_func2", -5, 5, 10],
        "F6": ["hybrid_func3", -5, 5, 10],
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
