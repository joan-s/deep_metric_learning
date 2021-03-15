# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:05:35 2018

@author: joans
"""

y = [ 0,  0,  1,  1,  2,  2,  2,  3 ,]
x = ['a','b','c','d','e','f','g','h',]
n=len(y)

triplets = [  [x[i], x[j], x[k]] 
              for i in range(n)
                for j in range(n)
                  for k in range(n)
                     if y[i]==y[j] and y[i] <> y[k] 
                        and i<>j and i<>k and j<>k
            ]