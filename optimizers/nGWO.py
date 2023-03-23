# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
"""

import random
import numpy
import math
from solution import solution
import time


def nGWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # Max_iter=1000
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5

    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    # initialize candidate position
    X_GWO = numpy.zeros(dim)
    X_DLH = numpy.zeros(dim)

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    # Initialize the fitness value for each search agents
    Fitness = numpy.zeros(SearchAgents_no)
    for i in range(0, SearchAgents_no):
        Fitness[i] = objf(Positions[i, :])
    
    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('nGWO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Pick fitness value for each search agents
            fitness = Fitness[i]

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        # Update the position of leaders using brownian motion (normal distribution)        
        Leader_pos = numpy.array([Alpha_pos, Beta_pos, Delta_pos])
        Leader_score = numpy.array([Alpha_score, Beta_score, Delta_score])
        
        for i in range(0, len(Leader_pos)):
            sigma = numpy.exp(
                (Leader_score[i] - Alpha_score) / abs(Alpha_score)
            )
            
            # define step size in each dimension
            step = numpy.random.randn(dim) * (sigma ** 2)
            par = 2 - 2 * ((l ** 2) / (Max_iter ** 2))
            
            old_pos = Leader_pos[i]
            new_pos = old_pos + par * step
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                new_pos[j] = numpy.clip(new_pos[j], lb[j], ub[j])
            
            if objf(new_pos) < objf(old_pos) :
                Leader_pos[i] = new_pos

        a = numpy.cos(numpy.pi / 2 * (l / Max_iter) ** 4)
        # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            
            # pick random wolf from population
            random_wolf = numpy.random.permutation(SearchAgents_no)
            
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                # Equation (3.3)
                C1 = 2 * r2
                # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha
                # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                # Equation (3.3)
                C2 = 2 * r2
                # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta
                # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                # Equation (3.3)
                C3 = 2 * r2
                # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta
                # Equation (3.5)-part 3

                X_GWO[j] = (X1 + X2 + X3) / 3  # Equation (3.7)
                
                # Return back each dimension of X_GWO that go beyond the boundaries of the search space
                X_GWO[j] = numpy.clip(X_GWO[j], lb[j], ub[j])
                
            # Calculate objective function for X_GWO
            Fit_GWO = objf(X_GWO)
            
            # Construct neighborhood for each search agents
            radius = numpy.sqrt(numpy.sum((Positions[i, :] - X_GWO)**2))
            neighbor_dist = numpy.array([numpy.sqrt(numpy.sum((Positions[i, :] - Positions[k, :])**2)) for k in range(SearchAgents_no)])
            neighbor_id = numpy.where(neighbor_dist <= radius)[0] # Equation (12)
            random_neighbor_id = numpy.random.randint(len(neighbor_id), size=dim)

            for j in range(dim):
                X_DLH[j] = Positions[i, j] + numpy.random.rand() * (
                    Positions[neighbor_id[random_neighbor_id[j]], j] 
                    - Positions[random_wolf[i], j]
                )  # Equation (12)
                
                # Return back each dimension of X_DLH that go beyond the boundaries of the search space
                X_DLH[j] = numpy.clip(X_DLH[j], lb[j], ub[j])
            
            # Calculate objective function for X_DLH
            Fit_DLH = objf(X_DLH)
            
            # select best candidate solution
            if Fit_GWO < Fit_DLH:
                temporary_pos = X_GWO.copy()
                temporary_fit = Fit_GWO.copy()
            else:
                temporary_pos = X_DLH.copy()
                temporary_fit = Fit_DLH.copy()
                
            if temporary_fit < Fitness[i]:
                Positions[i, :] = temporary_pos.copy()
                Fitness[i] = temporary_fit.copy()

                
        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "nGWO"
    s.objfname = objf.__name__

    return s
