#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Grey + RGB decomposition 

Gray level images from color images: 
Average & Luminance

@author: Ignacio Cordova Pou
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import skin
from skimage.data import astronaut

# Read
img = plt.imread('strawberrie.jpeg')


#------------------------------------
#Grey-scale inputs 2d array: 
    
imR = img[:,:,0]  #only R channel 
imG = img[:,:,1]  #only G channel 
imB = img[:,:,2]  #only B channel

#plotting in grey-scale
plt.figure(1)
plt.subplot(2,3,1)
plt.title('Extracció R')
plt.imshow(imR,cmap = 'gray')
plt.axis('off')

plt.subplot(2,3,2)
plt.title('Extracció G')
plt.imshow(imG,cmap = 'gray')
plt.axis('off')

plt.subplot(2,3,3)
plt.title('Extracció B')
plt.imshow(imB,cmap = 'gray')
plt.axis('off')

#------------------------------------
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
plt.title('Canal R')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(green)
plt.title('Canal G')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(blue)
plt.title('Canal B')
plt.axis('off')

#------------------------------------
#Gray scale images from color images 

#1. AVERAGE
"""Theoretically, the formula is 100% correct. But when 
writing code, you may encounter uint8 overflow error — 
the sum of R, G, and B is greater than 255. To avoid the 
exception, R, G, and B should be calculated respectively.

Grayscale = R / 3 + G / 3 + B / 3.

The average method is simple but doesn’t work as well 
as expected. The reason being that human eyeballs react 
differently to RGB. Eyes are most sensitive to green light,
 less sensitive to red light, and the least sensitive to 
 blue light. Therefore, the three colors should have
 different weights in the distribution.
"""

plt.figure(2)
av_im=imR/3 + imG/3 + imB/3 #imR,imG,imB are 2d arrays

plt.subplot(1,2,1)
plt.imshow(av_im,cmap='gray')
plt.title('Average Gray Scale')
plt.axis('off')

#2.LUMINANCE
"""The weighted method, also called luminosity method,
 weighs red, green and blue according to their wavelengths.
 The improved formula is as follows:

Grayscale  = 0.299R + 0.587G + 0.114B
"""
l_im = 0.299*imR+0.587*imG+0.114*imB

plt.subplot(1,2,2)
plt.imshow(l_im,cmap='gray')
plt.title('Weighted Gray Scale (Luminance)')
plt.axis('off')



#Implement a function for GAMMA CONTRAST
#Apartado i)
imm = img**0.5
imm = imm/np.max(imm)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(imm)

#Apartado ii)
imm = img**2.0

imm = imm/np.max(imm)
plt.subplot(1,2,2)
plt.imshow(imm)



#IMAGE COMPARISON


plt.figure()
im1 = astronaut()
print(np.max(im1))
plt.subplot(1,2,1)
plt.imshow(im1)



