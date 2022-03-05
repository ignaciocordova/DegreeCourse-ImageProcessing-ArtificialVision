#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:49:29 2022

@author: codefrom0
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import scipy as sc

im = data.astronaut()

imR = im[:,:,0]
imR2 = im[0:512:1, 
          0:512:1, 0]

plt.figure()
plt.imshow(imR, 
           cmap='gray') #colormap! 
plt.figure()
plt.imshow(imR2, 
           cmap='gray')