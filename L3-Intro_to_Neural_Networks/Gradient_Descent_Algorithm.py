#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:03:21 2019

@author: ferrarostefano
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plotting functions for both points and lines
def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)

# Activation (sigmoid) function, Continuous function 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Prediction function based on the sigmoid function (continuos output)
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

# Error function based on the Cross-Entropy for a 2 dimension set
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

# Perceptron weights/bias update, based on the Gradient Descent
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias

# Training Algorithm
def train(features, targets, epochs, learnrate, graph_lines=False):
    
    #Arrays init
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    
    # Iterations
    for e in range(epochs):
        for x, y in zip(features, targets):

            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        out = output_formula(features, weights, bias)
        
        # log-loss error on the entire training set (Continuos Error)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        
        # Printing Train log-loss error data
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                #If loss increase (every epochs/100 steps) we have an alert 
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            
            # Accuracy on the entire training set (Discrete Error)
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        
        # Plotting Boundary lines every epochs/10 steps
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data points
    plot_points(features, targets)
    plt.show()

    # Plotting the error line
    plt.figure()
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()

# Data extraction
data = pd.read_csv('DataSet1.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])


# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

# Input Parameter, Epochs represent the number of updates/iterations, Learning Rate the step
# between consecutive updates 
epochs = 1000
learnrate = 0.01

plt.figure()

# Main Program execution    
train(X, y, epochs, learnrate, True)