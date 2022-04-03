#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:55:00 2022

@author: Ignacio Cordova 
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import skimage




"""
1. Binarize an image using the error-diffusion algorithm. 
Split the resulting image using the procedure explained above.

    Jarvis-Judice-Ninke (JJN) algorithm for error compensation 
    
    imagen array 2D (gray scale) NORMALIZED to 1 !!!  
    """
    
imL =  0.299*np.double(skimage.data.astronaut()[:,:,0])+\
      0.587*np.double(skimage.data.astronaut()[:,:,1])+\
      0.114*np.double(skimage.data.astronaut()[:,:,2])

#normalized astronaut
ast = imL/imL.max()          

for ii in range(1,510): #no queremos recorrer los bordes 
    for jj in range(1,510):
        
        #definimos el error dependiendo de si se transforma a blanco o negro
        px = ast[ii,jj]
        if px>0.5:
            error = px-1
            
            #podríem aquí mateix modificar el píxel 
            #ast[i,j]=1
            
        else:
            error = px
            #ast[i,j]=0
            
        
        error = error /48 
        ast[ii,jj+1] += 7.*error
        ast[ii,jj+2] += 5.*error
        ast[ii+1,jj-2] +=3.*error 
        ast[ii+1,jj-1] +=5.*error 
        ast[ii+1,jj-0] +=7.*error 
        ast[ii+1,jj-0] +=5.*error 
        ast[ii+1,jj+1] +=3.*error 
        ast[ii+2,jj-2] +=1.*error 
        ast[ii+2,jj-1] +=3.*error 
        ast[ii+2,jj-0] +=5.*error 
        ast[ii+2,jj-0] +=3.*error 
        ast[ii+2,jj+1] +=1.*error


imJJN = ast>0.5


#Generate the split images A1 and A2 
A1 = np.random.choice([0, 1], size=(ast.shape[0],ast.shape[1]))

A2 = A1 ^ imJJN

#Recovered image:
B = A1^A2

ssim = skimage.metrics.structural_similarity(imJJN,B)


plt.figure()

plt.subplot(411)
plt.imshow(imJJN,cmap='gray')
plt.axis('off')

plt.subplot(412)
plt.imshow(A1,cmap='gray')
plt.axis('off')

plt.subplot(413)
plt.imshow(A2,cmap='gray')
plt.axis('off')

plt.subplot(414)
plt.title('ssim='+str(round(ssim,3)))
plt.imshow(B,cmap='gray')
plt.axis('off')

#%% Gray level image 

im = imL.astype('uint8')
imbin = np.unpackbits(im).reshape(im.shape[0],im.shape[1],8)

#binary bitplains (B matrices in the XOR method)
imbin0 = imbin[:,:,0]
imbin1 = imbin[:,:,1]
imbin2 = imbin[:,:,2]
imbin3 = imbin[:,:,3]
imbin4 = imbin[:,:,4]
imbin5 = imbin[:,:,5]
imbin6 = imbin[:,:,6]
imbin7 = imbin[:,:,7]

#A1 matrices of each bitplane
A01 = np.random.choice([0, 1], size=(im.shape[0],im.shape[1]))
A11 = np.random.choice([0, 1], size=(im.shape[0],im.shape[1]))
A21 = np.random.choice([0, 1], size=(im.shape[0],im.shape[1]))
A31 = np.random.choice([0, 1], size=(im.shape[0],im.shape[1]))
A41 = np.random.choice([0, 1], size=(im.shape[0],im.shape[1]))
A51 = np.random.choice([0, 1], size=(im.shape[0],im.shape[1]))
A61 = np.random.choice([0, 1], size=(im.shape[0],im.shape[1]))
A71 = np.random.choice([0, 1], size=(im.shape[0],im.shape[1]))

#A2 matrices of each bitplane
A02 = A01^imbin0
A12 = A11^imbin1
A22 = A21^imbin2
A32 = A31^imbin3
A42 = A41^imbin4
A52 = A51^imbin5
A62 = A61^imbin6
A72 = A71^imbin7

A1_combined = 128 * A01 + \
            64 * A11 + \
            32 * A21 +\
            16 * A31 + \
            8 * A41 + \
            4 * A51 + \
            2 * A61 + \
            A71
        
A2_combined = 128 * A02 + \
            64 * A12 + \
            32 * A22 +\
            16 * A32 + \
            8 * A42 + \
            4 * A52 + \
            2 * A62 + \
            A72
            
recuperada = A1_combined^A2_combined

ssim = skimage.metrics.structural_similarity(im,recuperada)


plt.figure()

plt.subplot(411)
plt.imshow(im,cmap='gray')
plt.axis('off')

plt.subplot(412)
plt.imshow(A1_combined,cmap='gray')
plt.axis('off')

plt.subplot(413)
plt.imshow(A2_combined,cmap='gray')
plt.axis('off')

plt.subplot(414)
plt.title('ssim='+str(round(ssim,3)))
plt.imshow(recuperada,cmap='gray')
plt.axis('off')





