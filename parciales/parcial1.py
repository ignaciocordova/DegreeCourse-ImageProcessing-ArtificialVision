#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:41:04 2022

@author: Ignacio Córdova 
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import skimage
import random 
from scipy.ndimage.measurements import histogram


im =  0.299*np.double(skimage.data.chelsea()[:,:,0])+\
      0.587*np.double(skimage.data.chelsea()[:,:,1])+\
      0.114*np.double(skimage.data.chelsea()[:,:,2])
      
hist_im = histogram(im, 0, 255, 256)/(300.*451.)

plt.figure()
plt.tight_layout()
plt.subplot(421)
plt.axis('off')
plt.imshow(im,cmap='gray')

plt.subplot(422)
plt.plot(hist_im)



#matrix probability that no photon reaches each pixel
probs = np.exp((-im*1000.)/im.sum()) 


#image starts all black 
pc = np.zeros([300,451])


#%% 300 frames

for frame in range(300):
    th = np.random.uniform(0.0,1.0,size=(300,451))
    
    #every pixel with prob<th receives a photon
    photons = probs<th
    pc = pc+photons


hist_1 = histogram(pc, 0, 255, 256)/(300.*451.)
ssim1 = skimage.metrics.structural_similarity(im,pc)

plt.subplot(423)
plt.title(round(ssim1,4))
plt.imshow(pc,cmap='gray')
plt.axis('off')

plt.subplot(424)
plt.plot(hist_1)




#%% 3000 frames

pc = np.zeros([300,451])

for frame in range(3000):
    th = np.random.uniform(0.0,1.0)
    
    #every pixel with prob<th receives a photon
    photons = probs<th
    pc = pc+photons


hist_1 = histogram(pc, 0, 255, 256)/(300.*451.)
ssim2 = skimage.metrics.structural_similarity(im,pc)


plt.subplot(425)
plt.title(round(ssim2,4))
plt.imshow(pc,cmap='gray')
plt.axis('off')

plt.subplot(426)
plt.plot(hist_1)
                
#%% 30000 frames

pc = np.zeros([300,451])

for frame in range(30000):
    th = np.random.uniform(0.0,1.0)
    
    #every pixel with prob<th receives a photon
    photons = probs<th
    pc = pc+photons


hist_1 = histogram(pc, 0, 255, 256)/(300.*451.)
ssim3 = skimage.metrics.structural_similarity(im,pc)

plt.subplot(427)
plt.title(round(ssim3,4))
plt.imshow(pc,cmap='gray')
plt.axis('off')

plt.subplot(428)
plt.plot(hist_1)