#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:27:39 2019

@author: ferrarostefano
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed()

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 100):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
    
# Data extraction
data = pd.read_csv("DataSet1.csv")
X = data[["x1","x2"]] 
y = data["y"] 

#Data conversion from DataFrame to np.array
X = pd.DataFrame(X).to_numpy()
y = pd.DataFrame(y).to_numpy()

boundary_lines = trainPerceptronAlgorithm(X,y)

#%% Plotting

plt.figure()

# X separation based on label(y)
condition = (y[:] == 0)
condition = np.squeeze(condition)
X_neg = X[condition]
X_pos= X[np.invert(condition)]

#Lines plot
x = np.arange(0,1.1,0.5) 

    # green lines for the iterations
for i in range(len(boundary_lines)):
    plt.plot(x,boundary_lines[i][0][0]*x+boundary_lines[i][1][0], 'g--', linewidth=0.5)

#black line for the last
plt.plot(x,boundary_lines[i][0][0]*x+boundary_lines[i][1][0], 'k-', linewidth=1.5)

#Points plot
plt.plot(X_pos[:,0],X_pos[:,1], 'bo')
plt.plot(X_neg[:,0],X_neg[:,1], 'ro')

plt.axis([0, 1, 0, 1])

    

