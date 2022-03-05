#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:45:04 2022

@author: codefrom0
"""



import matplotlib.pyplot as plt
import numpy as np
from skimage import data

im = data.retina()

#Build the greyscale images of the RGB channels

imR = im[:,:,0]
imG = im[:,:,1]
imB = im[:,:,2]

#Build each channel of RGB 
zeros = np.zeros((1411,1411,3),dtype='uint8')

imRR = zeros.copy()
imRR[:,:,0]=imR

imGG = zeros.copy()
imGG[:,:,1]=imG

imBB = zeros.copy()
imBB[:,:,2]=imB


#Plotting each image
plt.figure()

plt.subplot(2,3,1)
plt.imshow(imR,cmap='gray')

plt.subplot(2,3,2)
plt.imshow(imG,cmap='gray')

plt.subplot(2,3,3)
plt.imshow(imB,cmap='gray')

plt.subplot(2,3,4)
plt.imshow(imRR)

plt.subplot(2,3,5)
plt.imshow(imGG)

plt.subplot(2,3,6)
plt.imshow(imBB)

plt.tight_layout()


"RGB average"

im = plt.imread('strawberrie.jpeg')
imR = im[:,:,0]
imG = im[:,:,1]
imB = im[:,:,2]

im_avg =((1./3.)*imR+(1./3.)*imG+(1./3.)*imB)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(im_avg,cmap='gray')
plt.title('Simple average M=0.333*(R+G+B)')
plt.axis('off')

"Weighted average L=0.299*R+0.587*G+0.114*B"

im_L = ((0.299)*imR+(0.587)*imG+(0.114)*imB)

plt.subplot(1,2,2)
plt.imshow(im_L,cmap='gray')
plt.title('Weighted average L=0.299*R+0.587*G+0.114*B')
plt.axis('off')

"Weighted average with different colour maps"
plt.figure()

plt.subplot(1,3,1)
plt.imshow(im_L,cmap='magma')

plt.subplot(1,3,2)
plt.imshow(im_L,cmap='inferno')

plt.subplot(1,3,3)
plt.imshow(im_L,cmap='plasma')