#!/usr/bin/env python
# coding: utf-8

# # Neural networks with PyTorch
# 
# Deep learning networks tend to be massive with dozens or hundreds of layers, that's where the term "deep" comes from. You can build one of these deep networks using only weight matrices as we did in the previous notebook, but in general it's very cumbersome and difficult to implement. PyTorch has a nice module `nn` that provides a nice way to efficiently build large neural networks.

# In[2]:


# Import necessary packages

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import torch

import helper

import matplotlib.pyplot as plt

# In[4]:


### Run this cell

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# In[7]:


print(trainloader)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

# In[6]:


plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

# In[ ]:

def activation(x):

    return 1/(1+torch.exp(-x))

## Your solution

# output of your network, should have shape (64,10)
torch.manual_seed(7)

images = images.reshape(64,-1)

input_shape = 784
hidden_units = 256
output_shape = 10

W1 = torch.randn(input_shape, hidden_units)
W2 = torch.randn(hidden_units, output_shape)

B1 = torch.randn(hidden_units)
B2 = torch.randn(output_shape)
 
y = activation(torch.mm(images,W1) + B1)
out = torch.mm(y,W2) + B2

# In[ ]:


def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1,1)

# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))

# In[ ]:


from torch import nn
import torch.nn.functional as F

# In[ ]:


class Network(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

# In[ ]:


# Create the network and look at it's text representation
model = Network()
model

# In[ ]:

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        # Output layer, 10 units - one for each digit
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.ReLU(self.fc1(x))
        # Output layer with softmax activation
        x = F.ReLU(self.fc2(x))
        
        x = F.doftmax(self.fc3(x), dim=1) 
        
        return x

model = Network()
model

# In[ ]:


print(model.fc1.weight)
print(model.fc1.bias)


# For custom initialization, we want to modify these tensors in place. These are actually autograd *Variables*, so we need to get back the actual tensors with `model.fc1.weight.data`. Once we have the tensors, we can fill them with zeros (for biases) or random normal values.

# In[ ]:


# Set biases to all zeros
model.fc1.bias.data.fill_(0)


# In[ ]:


# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)


# ### Forward pass
# 
# Now that we have a network, let's see what happens when we pass in an image.

# In[ ]:


# Grab some data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)


# As you can see above, our network has basically no idea what this digit is. It's because we haven't trained it yet, all the weights are random!
# 
# ### Using `nn.Sequential`
# 
# PyTorch provides a convenient way to build networks like this where a tensor is passed sequentially through operations, `nn.Sequential` ([documentation](https://pytorch.org/docs/master/nn.html#torch.nn.Sequential)). Using this to build the equivalent network:

# In[ ]:


# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)


# Here our model is the same as before: 784 input units, a hidden layer with 128 units, ReLU activation, 64 unit hidden layer, another ReLU, then the output layer with 10 units, and the softmax output.
# 
# The operations are available by passing in the appropriate index. For example, if you want to get first Linear operation and look at the weights, you'd use `model[0]`.

# In[ ]:


print(model[0])
model[0].weight


# You can also pass in an `OrderedDict` to name the individual layers and operations, instead of using incremental integers. Note that dictionary keys must be unique, so _each operation must have a different name_.

# In[ ]:


from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model


# Now you can access layers either by integer or the name

# In[ ]:


print(model[0])
print(model.fc1)


# In the next notebook, we'll see how we can train a neural network to accuractly predict the numbers appearing in the MNIST images.
