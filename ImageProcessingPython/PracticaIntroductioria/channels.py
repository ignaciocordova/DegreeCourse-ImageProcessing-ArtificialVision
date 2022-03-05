#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:27:23 2022

Grey + RGB decomposition 

@author: Ignacio Cordova Pou
"""
import matplotlib.pyplot as plt
import numpy as np

# Read
img = plt.imread('beach.jpeg')



#Grey-scale inputs 2d array: 
    
imR = img[:,:,0]  #only R channel 
imG = img[:,:,1]  #only G channel 
imB = img[:,:,2]  #only B channel

#plotting in grey-scale
plt.figure()
plt.subplot(2,3,1)
plt.title('Canal R')
plt.imshow(imR,cmap = 'gray')

plt.subplot(2,3,2)
plt.title('Canal G')
plt.imshow(imG,cmap = 'gray')

plt.subplot(2,3,3)
plt.title('Canal B')
plt.imshow(imB,cmap = 'gray')


# RGB inputs 3d-arrays

red= np.copy(img)
blue= np.copy(img)
green = np.copy(img)

red[:,:,1] = 0 #eliminates the other channels 
red[:,:,2] = 0

green[:,:,0] = 0
green[:,:,2] = 0 

blue[:,:,0] = 0
blue[:,:,1] = 0

plt.subplot(2,3,4)
plt.imshow(red)

plt.subplot(2,3,5)
plt.imshow(green)

plt.subplot(2,3,6)
plt.imshow(blue)


