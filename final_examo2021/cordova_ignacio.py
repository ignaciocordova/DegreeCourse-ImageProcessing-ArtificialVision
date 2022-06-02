#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:51:39 2022



@author: Ignacio Cordova 
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy.ndimage.measurements import histogram


def get_channels(rgb):
    """
    Separates the channels of an RGB image.

    Input: 3-channel RGB image
    Output: 3 gray-scale images
    """
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    return r,g,b


"Problem 1"

imr,img,imb = get_channels(data.astronaut())

def census(im):
    im_census = np.zeros((im.shape[0],im.shape[1]))
    
    for ii in range(im.shape[0]-1):
        for jj in range(im.shape[1]-1):
            im_census[ii,jj] = 128*(im[ii,jj]<im[ii-1,jj-1])+\
            64*(im[ii,jj]<im[ii-1,jj])+\
            32*(im[ii,jj]<im[ii-1,jj+1])+\
            16*(im[ii,jj]<im[ii,jj-1])+\
            8*(im[ii,jj]<im[ii,jj+1])+\
            4*(im[ii,jj]<im[ii+1,jj-1])+\
            2*(im[ii,jj]<im[ii+1,jj])+\
            (im[ii,jj]<im[ii+1,jj+1])
    
    return im_census.astype('int')

imr_census = census(imr)
img_census = census(img)
imb_census = census(imb)

#%%

plt.figure()

plt.subplot(231)
plt.axis('off')
plt.imshow(imr,cmap='gray')

plt.subplot(232)
plt.imshow(img,cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.imshow(imb,cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.imshow(imr_census,cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(img_census,cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(imb_census,cmap='gray')
plt.axis('off')

#%%
"Humming Distances"

def humming_distance(im1,im2):
    return np.abs(((im1==im2)-1).sum())
            
print('The H distance between CTR and CTG is',
      humming_distance(img_census,imr_census))
print('The H distance between CTR and CTb is',
      humming_distance(imr_census,imb_census))
print('The H distance between CTB and CTG is',
      humming_distance(imb_census,img_census))


#%%
"Problem 2"

cat = plt.imread('Grayscale_Cat.jpeg')
dcat = plt.imread('Dithered_Cat.jpeg')


h = histogram(cat,0,255,256)
dh =  histogram(dcat,0,255,256)

plt.figure()
plt.subplot(221)
plt.axis('off')
plt.title('Original Image')
plt.imshow(cat,cmap='gray')

plt.subplot(222)
plt.axis('off')
plt.title('Dithered Image')
plt.imshow(dcat,cmap='gray')


plt.subplot(223)
plt.plot(h)

plt.subplot(224)
plt.plot(dh)


#%%


U, V = np.meshgrid(np.linspace(-1, 1, dcat.shape[1]), np.linspace(-1, 1, dcat.shape[0]))

A = 50
Gf = np.exp(-A*(U * U + V * V))


dcat_tf = np.fft.fftshift(np.fft.fft2(dcat))

filtered_dcat = np.abs(np.fft.ifft2(np.fft.ifftshift(dcat_tf * Gf))).astype('int')
filt_hist = histogram(filtered_dcat,0,255,256)


plt.figure()

plt.subplot(131)
plt.imshow(Gf,cmap='gray')

plt.subplot(132)
plt.imshow(filtered_dcat,cmap='gray')

plt.subplot(133)
plt.plot(dh,label='Dithered Car')
plt.plot(filt_hist,c='r',label='Filtered')








