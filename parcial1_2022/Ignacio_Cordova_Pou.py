#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:49:21 2022

@author: Ignacio Cordova Pou niub20082705
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import skimage
from scipy.ndimage.measurements import histogram
from scipy import signal
import scipy

#the image appears to be in the RGB space! 
im0 = plt.imread('virus.jpg')

imL =  0.299*np.double(im0[:,:,0])+\
      0.587*np.double(im0[:,:,1])+\
      0.114*np.double(im0[:,:,2])

im = imL.astype('uint8')


#%% Part 1


h = histogram(im,0,255,256)
hc = np.cumsum(h)/(im.shape[0]*im.shape[1])


plt.figure(constrained_layout=True,figsize=(20,20))

plt.subplot(331)
plt.imshow(im,cmap='gray')
plt.title('Original')
plt.axis('off')


plt.subplot(332)    
plt.plot(h)
plt.title('Histogram of original image')

plt.subplot(333)    
plt.plot(255*hc)
plt.title('Histogram Cumulative Sum Normalized of Original Image')



#%% Part 2 

im_eq = hc[im]
   
h_im_eq = histogram(im_eq,0,1,256)
h_im_eq_c = np.cumsum(h_im_eq)/(im.shape[0]*im.shape[1])

plt.subplot(334)
plt.imshow(im_eq,cmap='gray')
plt.title('Equalized image')
plt.axis('off')

plt.subplot(335)    
plt.plot(h_im_eq)
plt.title('Histogram of Equalized Image')

plt.subplot(336)    
plt.plot(255*h_im_eq_c)
plt.title('Histogram Cumulative sum of Equalized Image')


#%% Part 3: Locally equalized image

# im.reshape((8,8)) can't slice it

final = np.empty([im.shape[0],im.shape[1]])

#number of vertical and horizontal slices 
h_slices = int(im.shape[0]/8)
v_slices = int(im.shape[1]/8)


for i in range(h_slices+1):
    for j in range(v_slices+1):
        
        #equalize each slice
        slice_im = im[i*8:(i+1)*8,j*8:(j+1)*8]
        
        #using the histogram for each individual slice
        h = histogram(slice_im,0,255,256)
        hc = np.cumsum(h)/(slice_im.shape[0]*im.shape[1])

        im_eq = hc[slice_im]
        
        #writing the result in an empty 2D array
        final[i*8:(i+1)*8,j*8:(j+1)*8] =+ im_eq
        

h = histogram(final,0,final.max(),256)
hc = np.cumsum(h)/(final.shape[0]*im.shape[1])

plt.subplot(337)
plt.imshow(final,cmap='gray')
plt.title('LEI')
plt.axis('off')

plt.subplot(338)    
plt.plot(h)
plt.title('Histogram of Equalized Image')

plt.subplot(339)    
plt.plot(255*hc)
plt.title('Histogram Cumulative sum of Equalized Image')



#%% Part 4 


matriz_unos = np.ones([9,9])

filtro =  scipy.signal.order_filter(im, matriz_unos, 40)

im_orderfilter = im<=filtro

plt.figure()
#plotejem el filtre de order variable thresholding
plt.subplot(121)
plt.imshow(final,cmap='gray')
plt.title('LEI')
plt.axis('off')

im_orderfilter = im<=filtro

plt.subplot(122)
plt.imshow(im_orderfilter, cmap='gray')
plt.title('Binarization with order filter')
plt.axis('off')

ssim = skimage.metrics.structural_similarity(final,im_orderfilter)
print('The ssim index is',ssim)

"""
    I tried using a mask of a similar size as the 
    slices of LEI. I could not get a good structural
    similarity index. Visually the images look similar
    eventhough ssim is very low.
"""

#Comment to Part 5: 
"""
    Local equalization significantly improves the equalization
    of the image. The range of values used is much better
    distributed than with the previous method seen in 
    Figure_1. The cumulative histogram of the equalized image
    looks as good as it can get. 
"""



