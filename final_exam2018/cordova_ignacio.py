#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:11:25 2022

@author: Ignacio Cordova 
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy import ndimage
from skimage import restoration

from matplotlib import cm 

def luminance(rgb):
    """
    Calculates the luminance of an RGB image

    Input: 3-channel RGB image
    Output: Luminance image
    """
    l_im =  0.299*np.double(rgb[:,:,0])+\
            0.587*np.double(rgb[:,:,1])+\
            0.114*np.double(rgb[:,:,2])
    

    return l_im 

#creo la imatge amb escala de grisos
n=256
test = np.zeros((n,n))
for i in range(n):
    test[:,i] = i

#creo la imatge en jet
jetv = cm.jet(range(256)) 
jetr = jetv[:,0]
jetg = jetv[:,1]
jetb = jetv[:,2]

jet_rgb = np.zeros((n,n,3))
for i in range(n):
    jet_rgb[:,i,0]= jetv[i,0]
    jet_rgb[:,i,1]= jetv[i,1]
    jet_rgb[:,i,2]= jetv[i,2]
    

#creo la imatge en ver
virv = cm.viridis(range(256)) 
virr = virv[:,0]
virg = virv[:,1]
virb = virv[:,2]

vir_rgb = np.zeros((n,n,3))
for i in range(n):
    vir_rgb[:,i,0]= virv[i,0]
    vir_rgb[:,i,1]= virv[i,1]
    vir_rgb[:,i,2]= virv[i,2]
    

ljet = luminance(jet_rgb)
lvir = luminance(vir_rgb)



plt.figure()

plt.subplot(231)
plt.title('TEST')
plt.imshow(test,cmap='gray')
plt.subplot(232)
plt.title('Jet cm as RGB')
plt.imshow(jet_rgb)

plt.subplot(233)
plt.title('Viridis cm as RGB')

plt.imshow(vir_rgb)

plt.subplot(234)
plt.title('Row 50 of each Luminance')
plt.plot(test[50,:]/256,label='Test')
plt.plot(ljet[50,:],label='Jet')
plt.plot(lvir[50,:],label='Viridis')
plt.legend()
plt.subplot(235)
plt.title('Luminance LJET')
plt.imshow(ljet,cmap='gray')
plt.subplot(236)
plt.title('Luminance LVIR')
plt.imshow(lvir,cmap='gray')


#%% 

"Problema 2"


im = luminance(plt.imread('chlamydomonas.jpg'))

#filtre de suavitzat (tipus Canny)

C = (1/159)*np.array([[2,4,5,4,2],
     [4, 9, 12, 9, 4],
     [5, 12, 15, 12, 5],
     [4, 9, 12, 9, 4],
     [2, 4, 5, 4, 2]])

canny = ndimage.filters.convolve(im, C, 
                                        output=None, 
                                        mode='reflect', 
                                        cval=0.0, 
                                        origin=0)


chamb = restoration.denoise_tv_chambolle(im,weight=120)








plt.figure()
plt.subplot(131)
plt.title('Original')
plt.imshow(im,cmap='gray')
plt.subplot(132)
plt.title('Canny filter convolution')
plt.imshow(canny,cmap='gray')

plt.subplot(133)
plt.title('Chambolle (skimage)')
plt.imshow(chamb,cmap='gray')



