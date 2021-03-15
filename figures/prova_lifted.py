# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 11:38:56 2018

@author: joans
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


alpha = 1.0
np.random.seed(0)
n = 100
J = np.zeros((n,n))
Jtilde = np.zeros((n,n))
t = np.arange(0.0,1.0,1./n)
d1, d2 = np.meshgrid(t,t)
for i in range(n):
    for j in range(n):    
        J[i,j] = np.maximum(np.max(alpha-d1[i,j]),np.max(alpha-d2[i,j]))
        Jtilde[i,j] = np.log(np.sum(np.exp(alpha-d1[i,j])) + np.sum(np.exp(alpha-d2[i,j])))
# el sum no cal

plt.matshow(J,cmap=plt.cm.gray,interpolation='nearest')
plt.matshow(Jtilde,cmap=plt.cm.gray,interpolation='nearest')

fig = plt.figure()
ax = fig.gca(projection='3d')
X = d1
Y = d2
Z = J
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
Z = Jtilde
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
