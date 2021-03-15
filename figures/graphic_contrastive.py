# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:33:48 2018

@author: joans
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

contrastive = False
cosine = False
logistic = True

if contrastive:
    lastx = 7.
    d = np.arange(0, lastx,step=0.1)
    m = 5.
    c1 = d**2
    c2 = np.maximum(0., m-d)**2
    c3 = d
    c4 = np.maximum(0., m-d)
    c5 = d**2
    c6 = np.maximum(0., m**2 - d**2)
    #p = (1 + np.exp(-m))/(1+np.exp(d-m))
    #c5 = p*np.log(1+np.exp(d-m) )
    
    plt.figure()
    plt.plot(d, c1,'b',linewidth=2,label='same, $D^2$')
    plt.plot(d, c2, 'r', linewidth=2, label='different $(\max(0,m-D))^2$')
    plt.plot(d, c3,'b:',linewidth=2,label='same $D$')
    plt.plot(d, c4, 'r:', linewidth=2, label='different $\max(0,m-D)$')
    plt.plot(d, c5,'b--',linewidth=2,label='same $D^2$')
    plt.plot(d, c6, 'r--', linewidth=2, label='different $\max(0,m^2-D^2)$')
    #plt.plot(d, c5, 'g', linewidth=2, label='distance logistic')
    plt.plot([m,m],[0,lastx**2],'k:')
    plt.axis([0,lastx,-1,lastx**2])
    plt.xlabel('Euclidean distance D')
    plt.ylabel('Contrastive loss')
    plt.legend(loc='upper left')

if cosine:
    angle = np.arange(0.,0.5,0.01)    
    c1 = (1.0 - np.cos(angle*np.pi))**2
    c2 = (np.cos(angle*np.pi))**2
    plt.figure()
    plt.plot(angle, c1,'b',linewidth=2,label='same')
    plt.plot(angle, c2, 'r', linewidth=2, label='different')
    plt.legend(loc='middle left')
    plt.xlabel('angle in $\pi$ rads')
    plt.ylabel('cosine loss')
    
    
if logistic:
    lastx = 10.
    d = np.arange(0.1, lastx,step=0.01)
    m = lastx/2.
    prob = (1 + np.exp(-m)) / (1 + np.exp(d-m))


    plt.figure()
    plt.plot(d, prob,'b',linewidth=2)
    plt.plot([m,m],[0,1.0],'k:')
    plt.xlabel('Euclidean distance D')
    plt.ylabel('Probability of being similar')

    plt.figure()
    plt.plot(d, -np.log(prob),'b',linewidth=2,label='same')
    plt.plot(d, -np.log(1. - prob),'r',linewidth=2,label='different')
    plt.plot([m,m],[0,8.0],'k:')
    plt.xlabel('Euclidean distance D')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
   