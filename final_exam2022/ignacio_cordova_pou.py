#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:01:50 2022

Examen final PIVA 2022

@author: Ignacio Córdova 
"""

import matplotlib.pyplot as plt
import numpy as np 
from numpy import unravel_index
from skimage.filters.rank import entropy

import cordova_functions as cordova


#the images are saved in PNG format so I will use one
#channel for the gray level and the first 3 channels for RGB

im_gray = plt.imread('ct.png')[:,:,0]
im_rgb = plt.imread('mri_color.png')[:,:,0:3]

#%%
#_________Task 1 (0.5p)____________
"Prepare a 2x3 figure. Display gray-level and RGB."

plt.figure()

plt.subplot(231)
plt.title('Gray level image')
plt.axis('off')
plt.imshow(im_gray,cmap='gray')

plt.subplot(232)
plt.title('Original RGB image ')
plt.axis('off')
plt.imshow(im_rgb)


#%%
#_________Task 2 (2.0p)____________
"""
Calculate the relative shift among images. On the console, 
print the coordinates of the correlation maximum. Show the 
correlation distribution. 
"""

r,g,b = cordova.channels(im_rgb)

# I use the green channel "g" for the MRI image
corr = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(im_gray))* \
                                            np.conjugate(np.fft.fftshift(np.fft.fft2(g))))))

    
print('The coordinates of the correlation maximum are: ',
      unravel_index(corr.argmax(), corr.shape))


plt.subplot(233)
plt.title('Correlation distribution')
plt.axis('off')
plt.imshow(corr,cmap='jet')




#%%
#_________Task 3 (3.0p)____________
"Transform image mri_color.png to the HSV model"


def rgb_to_hsv(im):
    """
    Function that transforms an RGB image into the HSV model

    Parameters
    ----------
    im : 3-channel  RGB image

    Returns
    -------
    HSV : 3-channel HSV image

    """
    
    r,g,b = cordova.channels(im)
    HSV = np.zeros((im.shape[0],im.shape[1],3))
    H = np.zeros((im.shape[0],im.shape[1]))
    S = np.zeros((im.shape[0],im.shape[1]))
    
    V =  np.max(im,axis=2)
    m = np.min(im,axis=2)
    C = V - m 
    
    for ii in range(im.shape[0]):
        for jj in range(im.shape[1]):
            
            if C[ii,jj] != 0:
                if V[ii,jj] == r[ii,jj] :
                    H[ii,jj] = ((g[ii,jj]-b[ii,jj])/C[ii,jj])%6 
                if V[ii,jj] == g[ii,jj] : 
                    H[ii,jj] = (b[ii,jj]-r[ii,jj])/C[ii,jj] +2
                if V[ii,jj] == b[ii,jj] :
                    H[ii,jj] = (r[ii,jj]-g[ii,jj])/C[ii,jj] +4 
                
            if V[ii,jj] !=0:
                S[ii,jj] = C[ii,jj]/V[ii,jj]

    H = H/6.0
    
    HSV[:,:,0] = np.copy(H)
    HSV[:,:,1] = np.copy(S)
    HSV[:,:,2] = np.copy(V)
    
    return HSV


#%%

im_hsv = rgb_to_hsv(im_rgb)

plt.subplot(234)
plt.title('HSV image')
plt.axis('off')
plt.imshow(im_hsv)


#%%
#_________Task 4 (3.0p)____________

"""
Replace the V channel with the gray level image ct.png. 
Transform the modified HSV ensemble to RGB
"""

im_hsv[:,:,2] = np.copy(im_gray)
 
 
def f(HSV,n):
    """
    Function that given an HSV image computes f(n) 
    (see Appendix B)

    Parameters
    ----------
    HSV : 3-channel HSV image
    n : integer that will define which channel are 
    we retrieving (n= 5 for  R, 3 for G and 1 for B)

    Returns
    -------
    F : channel (2-d array)

    """
    H,S,V = cordova.channels(HSV)
    F = np.zeros((HSV.shape[0],HSV.shape[1]))
    k = (n+6*H)%6
    
    for ii in range(HSV.shape[0]):
        for jj in range(HSV.shape[1]):
            F[ii,jj] = V[ii,jj] \
                -V[ii,jj]*S[ii,jj]*max(0,min(k[ii,jj],4-k[ii,jj],1))

    return F 
    

#%%

fused_rgb = np.dstack((f(im_hsv,5),f(im_hsv,3),f(im_hsv,1)))


plt.subplot(235)
plt.title('New fused image')
plt.axis('off')
plt.imshow(fused_rgb)



#%%
#_________Task 5 (1.5p)____________
"""
Calculate the local entropy of the final RGB image calculated
 in the previous section using a 7x7 window. 

As we discussed in class, for color images, the entropy
should be calculated for every channel.
 """

R,G,B = cordova.channels(fused_rgb)

sR = entropy(R,np.ones([11,11],dtype=np.uint8))
sG = entropy(G,np.ones([11,11],dtype=np.uint8))
sB = entropy(B,np.ones([11,11],dtype=np.uint8))

"""I calculate the local entropy of the RGB image as the 
average between the three channels """

ent = (sG + sB + sR)/3.0

plt.subplot(236)
plt.title('Entropy of fused im')
plt.axis('off')
plt.imshow(ent)
plt.colorbar()





