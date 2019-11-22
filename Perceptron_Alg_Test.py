#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:59:54 2019

@author: ferrarostefano
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x1,w1,w2,b): 
    return(-w1/w2*x1-b/w2)
    
def CoeffAdd(w1,w2,b,LR):
    w1 = w1 - LR*P[0]
    w2 = w2 - LR*P[1]
    b = b - LR
    return(w1,w2,b)


LR = 0.1    

w1 = 4
w2 = 5
b = -11

P = [3,1]

x1 = np.arange(0.0, 10.0, 0.1)

plt.figure()
plt.plot(P[0],P[1], 'o')

for i in range(11):
    try:
        plt.plot(x1, f(x1,w1,w2,b))
        w1,w2,b = CoeffAdd(w1,w2,b,LR)
    except:
        w1,w2,b = CoeffAdd(w1,w2,b,LR)
        pass