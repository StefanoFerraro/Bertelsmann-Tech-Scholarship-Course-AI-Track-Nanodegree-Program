#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:03:21 2019

@author: ferrarostefano
"""

import numpy as np
import pandas as pd
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.io as pio

# Plotting function for both points and lines
def plot_routine(fig, data, errors, m, b):
    
    data['errors'] = errors
     
    x1 = np.arange(-1, 3, 1) 
    x2 = m*x1+b            
    
    data = [go.Scatter(mode='markers', visible=False, marker_color=data["colors"], marker=dict(size=data['errors'], sizeref = 0.025, 
                             line=dict(color='Black', width=1)), x=data["x1"], y=data["x2"],
                             hoverinfo="text", hovertemplate= "Cross-Entropy: %{marker.size:.3f}" + "<extra></extra>"),
            go.Scatter(mode="lines",visible=False, x = x1, y = x2, line=dict(color="black"))]
    
    fig.add_traces(data)
    
# Plotting function for the slider
def plot_slider(fig):
    
    steps= []
    
    # Define the steps dict, for each step of the slider we want just two dataset to be visible:
    # one boundary_line and the related set of points (with the related error)
    for i in range(int(len(fig.data)/2)):
        step = dict(method="restyle",args=["visible", [False] * len(fig.data)])
        
        step["args"][1][2*i] = True  # Toggle i'th trace to "visible"
        step["args"][1][2*i+1] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    
    active_level = 0    # Active Level of dataset when the plot is generated
    fig.data[active_level].visible = True
    fig.data[active_level+1].visible = True
    
    sliders = [dict(
    active= active_level,   # Slider starting level
    currentvalue={"prefix": "Iteration: "},
    pad={"t": 50}, 
    steps=steps)]
    
    # Updating the plot with the designed slider
    fig.update_layout(sliders=sliders)


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
def train(data, features, targets, epochs, learnrate, graph_lines=False):
    
    # Plot Init, layout settings
    layout = go.Layout(
    showlegend=False,
    title = "Gradient Descent Algorithm",
    template = pio.templates['plotly'],
    xaxis = dict(range=[-0.01, 1.01]),
    yaxis = dict(range=[-0.01, 1.01]))
    
    fig = go.Figure(layout=layout)
    
    #Arrays Init
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    
    # Iterations
    for e in range(epochs):
        
        # Weights and Bias update based on the entire data set 
        for x, y in zip(features, targets):
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        # Prediction function (outuput continuos value == propability to be classify right)
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
        
        # Plot Boundary lines and points + Error as Size every epochs/100 steps
        if graph_lines and e % (epochs / 100) == 0:
            
            output = output_formula(features, weights, bias)       
            point_errors = error_formula(targets,output)
            
            plot_routine(fig, data, point_errors, -weights[0]/weights[1], -bias/weights[1])
    
    # Slider implementation
    plot_slider(fig)
    
    #Plotting
    plot(fig)
    fig.show()

# Data extraction
data = pd.read_csv('DataSet1.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])

data.columns = ["x1","x2","y"]

# Associate '0' and '1' with 'red' and 'blue'
colors = []
for i in range(len(data['y'])):
    if data['y'][i] == 1:
       colors.append('red')
    else:
       colors.append('blue') 

data['colors'] = colors

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed()

# Input Parameter, Epochs represent the number of updates/iterations, Learning Rate the step
# between consecutive updates 
epochs = 100
learnrate = 0.01

# Main Program execution    
train(data, X, y, epochs, learnrate, True)