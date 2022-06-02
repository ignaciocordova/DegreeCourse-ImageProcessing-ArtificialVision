#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final exam, June 13th, 2019 PIVA, UB 

@author: Ignacio Cordova 
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy import ndimage
from tqdm import tqdm
from skimage.filters.rank import entropy
from skimage.morphology import disk

import cordova_functions as do 

im1 = do.luminance(data.astronaut()) 
im1r = data.astronaut()[:,:,0]
im1g = data.astronaut()[:,:,1]
im1b = data.astronaut()[:,:,2]

im2 = do.luminance(data.chelsea())

def get_hash(image):
    im = np.copy(image)
    im.resize((8,9))
    imb = im[:,0:8]<im[:,1:9]
    imhash = np.reshape(imb,(1,64))
    return imhash

def ham_dist(im1,im2):
    dif = abs(get_hash(im1)==get_hash(im2)-1)
    
    return dif.sum()

d = ham_dist(im1,im2)



#%%

im0 = plt.imread('ecolidn.png')
im = do.luminance(im0) #luminance of the defocused image
d = np.copy(im) # "d" remains intact through all iterations


#function that creates a mask with a circle of r pixels radius
def circle_mask(r):
    """
    Creates a mask with a circle of r pixels radius.

    Input: radius of the circle
    Output: mask
    """
    u, v = np.meshgrid(np.linspace(-1, 1, 2*r), np.linspace(-1, 1, 2*r))
    circ = (u**2 + v**2)<1
    return circ

psf = circle_mask(37)
plt.figure()
plt.title('Mask used for convolutions')
plt.imshow(psf)


#%%

#FIRST TRY (using convolutions)

#for i in tqdm(range(50)):
#    convo = ndimage.convolve(im,psf)
#    convo2 = ndimage.convolve(d/convo,psf)
#    im = im*np.abs(convo2)


#%%

#not very good results


plt.figure()
plt.subplot(121)
plt.imshow(d,cmap='gray')

plt.subplot(122)
plt.imshow(im,cmap='gray')


#%%

#second try (using products in Fourier Space)

def circle(r,im):
    """
    Creates a circle of radius r centered in the image im.

    Input: radius r and gray-scale image im
    Output: image with a circle of radius r centered in the image im
    """
    x,y = np.ogrid[0:im.shape[0],0:im.shape[1]]
    mask = (x-im.shape[0]/2)**2 + (y-im.shape[1]/2)**2 <= r**2
    im1 = im*mask
    return im1>0



psf = circle(37,im)
plt.figure()
plt.title('I will Fourier transform this to perform products')
plt.imshow(psf)
psf_tf = np.fft.fft2(psf)

#%%

for i in tqdm(range(50)):
    im_tf = np.fft.fft2(im)
    
    #producte entre i_k i h a l'espai de Fourier.
    convo = np.abs(np.fft.ifftshift(np.fft.ifft2(im_tf*psf_tf))) #i fem TF inversa
    
    #per poder fer una divisió normal
    ratio = d/convo 
    #tf per poder fer el segon producte a l'espai de Fourier
    ratio_tf = np.fft.fft2(ratio)
    
    #segon producte 
    convo2 = np.fft.ifftshift(np.fft.ifft2(ratio_tf*psf_tf))
    
    #multipliquem per el modul 
    im_recovered = im*np.abs(convo2)
    
    #aquesta s'utilitza per la següent iteració 
    im = np.copy(im_recovered)


#%%

#not very good results, or yes?...


plt.figure()
plt.subplot(121)
plt.imshow(d,cmap='gray')

plt.subplot(122)
plt.imshow(im_recovered,cmap='gray')


#%%

s1=entropy(d, disk(5)) #Calculem l'entropia de la imatge d
s2=entropy(im_recovered/im_recovered.max(), disk(5))#Calculem l'entropia de la imatge final.

#_________PLOTS:___________________________________________
plt.figure("Exercici 2, reconstruccio Eschericcia Coli")


plt.subplot(2,2,1)
plt.title("Imatge E. Coli desenfocada")
plt.imshow(d,cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Imatge E. Coli reconstruida")
plt.imshow(im_recovered, cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Entropia imatge desenfocada")
plt.imshow(s1, cmap='rainbow')
plt.axis('off')
plt.colorbar()

plt.subplot(2,2,4)
plt.title("Entropia imatge reconstruida")
plt.imshow(s2, cmap='rainbow')
plt.axis('off')
plt.colorbar()
