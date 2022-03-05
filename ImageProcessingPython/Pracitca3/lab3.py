#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:47:57 2022

Lab3 : Image binarization and Dithering

@author: codefrom0
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data 
import scipy as scipy
from scipy import signal


""" Binarization thresholding"""

page = data.page()

#and a float type between 0 and 1
im = page/page.max()


#we will try different thresholds
th = np.arange(0,1,0.1)
plotindx = 0

plt.figure(constrained_layout=True, figsize = (20, 20))

for t in th:
    plotindx = plotindx+1
    imb=im>t
    plt.subplot(2,5,plotindx)
    plt.imshow(imb,cmap='gray')
    plt.axis('off')
    plt.title(str(t)[0:3]+' threshold')
    


"""Adaptive thresholding """

plt.figure(constrained_layout=True, figsize = (20, 20))


#let's plot a binary global threshold to compare
t = 0.5
imb = im>t
plt.subplot(234)
plt.imshow(imb,cmap='gray')
plt.title('0.5 global threshold')
plt.axis('off')


#MEDIAN THRESHOLD

# media = scipy.signal.medfilt(im, [mask size])
# ordena los valores de la imagen dentro de mask y coge el del medio


media = scipy.signal.medfilt(im, [11,11])

#plotejem el filtre de median threshold
plt.subplot(232)
plt.imshow(media,cmap='gray')
plt.title('Median Threshold')
plt.axis('off')

im_medfilt=im<=media


plt.subplot(231)
plt.imshow(im,cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(235)
plt.title('Median Thresholding Binary')
plt.imshow(im_medfilt, cmap='gray')
plt.axis('off')




#VARIABLE THRESHOLD

#scipy.signal.order_filter(im, mask(matriz), rank)
# ordena los valores de la imagen dentro de mask y coge el nº rank 
#
#si la matriz tiene unos (1) y ceros (0) la "mask" solo usará
#los píxeles asociados a unos.
matriz_unos = np.ones([19,19])

filtro =  scipy.signal.order_filter(im, matriz_unos, 60)

#plotejem el filtre de order variable thresholding
plt.subplot(233)
plt.imshow(filtro,cmap='gray')
plt.title('Adaptive Threshold')
plt.axis('off')

im_orderfilter = im<=filtro

plt.subplot(236)
plt.imshow(im_orderfilter, cmap='gray')
plt.title('Adaptive Thresholding Binary')
plt.axis('off')





""" Dithering """

# We want to get a binary image with some compensation for the erros
# We compensate the error on one pixel with the pixels sorrounding it
# more conretely, the error is distributed in the down right pixels

astronaut = data.astronaut()
imint = astronaut[:,:,0]

ast = imint/imint.max()



"""Floyd-Steinbeirg (FS)"""
#debemos iterar para todos los píxeles


for i in range(1,511): #no queremos recorrer los bordes 
    for j in range(1,511):
        
        #definimos el error dependiendo de si se transforma a blanco o negro
        px = ast[i,j]
        if px>0.5:
            error = px-1
            
            #podríem aquí mateix modificar el píxel 
            #ast[i,j]=1
            
        else:
            error = px
            #ast[i,j]=0
        
        #una vez definido propagamos el error al resto de píxeles
        
        ast[i,j+1]=ast[i,j+1]     + (7./16.)*error
        ast[i+1,j+1]=ast[i+1,j+1] + (1./16.)*error
        ast[i+1,j]=ast[i+1,j]     + (5./16.)*error
        ast[i+1,j-1]=ast[i+1,j-1] + (3./16.)*error
            
            
imFS = ast>0.5



    
"""Jarvis-Judice-Ninke"""


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


imJNN = ast>0.5




plt.figure()

plt.subplot(121)
plt.imshow(imFS,cmap='gray')
plt.title('FS')
plt.axis('off')


plt.subplot(122)    
plt.imshow(imFS,cmap='gray')
plt.title('JNN')
plt.axis('off')




