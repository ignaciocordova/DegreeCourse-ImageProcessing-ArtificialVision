#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:24:17 2022

@author: Ignacio Cordova 
"""


import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage.measurements import histogram
from scipy import ndimage


im =  skimage.data.astronaut()[:,:,1]
h= histogram(im, 0, 255, 256)/(512.*512.)

suma = 0.0

for i in range(len(h)):
    if suma<0.5:
        suma += h[i]
    else:
        print('TB = ',i)
        break

im_bin = im>i 
ssim1 = skimage.metrics.structural_similarity(im,im_bin)

otsu = im>skimage.filters.threshold_otsu(im)
ssim2 = skimage.metrics.structural_similarity(im,im_bin)


imM = ndimage.generic_filter(im, np.mean, footprint=np.ones([25,25]) )
im_adaptive = im>imM
ssim3 = skimage.metrics.structural_similarity(im,im_adaptive)

plt.figure()

plt.subplot(131)
plt.imshow(im_bin,cmap='gray')
plt.title('Histogram TB threshold 108, ssim='+str(round(ssim1,3)))

plt.subplot(132)
plt.imshow(otsu,cmap='gray')
plt.title('Otsu threshold 103, ssim=,'+str(round(ssim2,3)))

plt.subplot(133)
plt.imshow(im_bin,cmap='gray')
plt.title('Adaptive threshold, ssim='+str(round(ssim3,3)))







