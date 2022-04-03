#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:47:57 2022

Lab # 3: Image binarization.
3.1. Adaptive thresholding.
3.2. Error diffusion binarization (dithering) FS & JNN. 
3.3. Color dithering. The HSV color model.

@author: codefrom0
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from skimage import data 
import scipy as scipy
from scipy import signal
import cordova_functions as cordova


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
    


"""3.1 Adaptive thresholding """

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





""" 3.2 Dithering """

# We want to get a binary image with some compensation for the erros
# We compensate the error on one pixel with the pixels sorrounding it
# more conretely, the error is distributed in the down right pixels

astronaut = data.astronaut()
imint = astronaut[:,:,0]

ast = imint/imint.max()



"""Floyd-Steinbeirg (FS)"""
#debemos iterar para todos los píxeles


imFS = cordova.FS(ast)



    
"""Jarvis-Judice-Ninke"""


imJJN = cordova.JJN(ast)




plt.figure()

plt.subplot(121)
plt.imshow(imFS,cmap='gray')
plt.title('FS')
plt.axis('off')


plt.subplot(122)    
plt.imshow(imFS,cmap='gray')
plt.title('JNN')
plt.axis('off')



"3.3. Color dithering. The HSV color model."

#Imatge RGB amb valors [0,1]
im0 = data.astronaut()
astronaut = im0/im0.max()

#Passem la imatge (ha de tenir valors [0,1]) a format HSV
imhsv = matplotlib.colors.rgb_to_hsv(astronaut)


#Farem una binarització només del canal V (l'últim!!!)
#el copiem en una matriu per fer-hi feina.
imv = np.copy(imhsv[:,:,2])

imvFS = cordova.FS(imv)

#Substituïm a la imatge HSV d’abans el canal V antic pel nou, binaritzat
imhsv[:,:,2] = np.copy(imvFS)

#Recuperem el format RGB per poder representar-la
imrgb = matplotlib.colors.hsv_to_rgb(imhsv)




#RGB is between 0 and 1
test1 = np.rint(imrgb*5)
#test1 is floats rounded to nearest
#integer resulting in [0.,1.,2.,3.,4.] 
test2 = test1/5
#test2 is floats [0.,0.2,0.4,0.6,0.8]




imbin6c = np.float64(np.rint(imrgb*5)/5)
 
plt.figure()

plt.subplot(121)
plt.imshow(imrgb)
plt.title('FS to V channel in HSV')
plt.axis('off')


plt.subplot(122)    
plt.imshow(imbin6c)
plt.title('np.rint[RGB*5]/5')
plt.axis('off')




#Color reduction from 256x256x256 to 6x6x6

test_im = data.coffee()
im6x6x6 = cordova.color_reduction(test_im)



plt.figure(figsize = (20, 20))

plt.subplot(121)
plt.imshow(test_im)
plt.title('Original')
plt.axis('off')


plt.subplot(122)    
plt.imshow(im6x6x6)
plt.title('[0,43,86,...,215]')
plt.axis('off')




