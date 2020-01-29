#!/usr/bin/env python
# coding: utf-8

# In[2]:


# First, import PyTorch
import torch

# In[3]:


def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

# In[4]:


### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 5 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

# In[5]:


## Calculate the output of this network using the weights and bias tensors
sumation = torch.sum(weights*features)+bias
activation(sumation)

# In[19]:


## Calculate the output of this network using matrix multiplication

#weights.reshape(5,1)
activation(torch.mm(features,weights.view(5,1))+bias)

# In[21]:


### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

# In[26]:


## Your solution here
h1_2 = activation(torch.mm(features,W1) + B1)
y = torch.mm(h1_2,W2) + B2
activation(y)
