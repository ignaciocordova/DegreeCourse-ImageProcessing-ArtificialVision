#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:00:09 2022

@author: codefrom0
"""

"""
Lab #6: Defocused images and image restoration filters.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import numpy.random as random
import scipy.stats as stat


im = data.camera()

"6.1. Building defocused image as convolution d = i*|PSF|^2"

#PSF obtained from a circle of radius R=0.03:

x = np.linspace(-1, 1, 512)
X, Y = np.meshgrid(x, x)
d = np.sqrt(X * X + Y * Y)
circ = d < 0.1


#Ahora hago la transformada de fourier de todo esto para obtener el operador que
#me convertirá la imagen enfocada en una desenfocada con el producto de convolución

psf = np.abs(np.fft.fftshift(np.fft.fft2(circ)))
#Tiene que ir al cuadrado
psf = psf**2
#Observo que el filtro no está normalizado, lo normalizo y enchufo todo como uint8
psf = 255*(psf/np.max(psf))


#Ahora hago el producto de convolución que en el espacio
#de Fourier es una simple multiplicación 

im_tf = np.fft.fftshift(np.fft.fft2(im))
psf_tf = np.fft.fftshift(np.fft.fft2(psf))

prod_convo = np.abs(np.fft.ifftshift(np.fft.ifft2(im_tf*psf_tf)))

#Ahora hago la transformada inversa y normalizo
im_desenfocada = np.uint8(255 * prod_convo / prod_convo.max())


plt.figure()

plt.subplot(321)
plt.title('Original Image')
plt.imshow(im,cmap='gray')
plt.axis('off')

plt.subplot(322)
plt.title('Circle')
plt.imshow(circ,cmap='gray')
plt.axis('off')

plt.subplot(323)
plt.imshow(im,cmap='gray')
plt.axis('off')

plt.subplot(324)
plt.title('|PSF|^2 = FT(circle)^2')
plt.imshow(psf,cmap='gray')
plt.axis('off')


plt.subplot(325)
plt.title('d = i*|PSF|^2')
plt.imshow(im_desenfocada,cmap='gray')
plt.axis('off')


#%%

"""6.2 6.2. Image restoration filters.
Tenemos la imagen desenfocada y el PSF y queremos recuperarla"""

"INVERSE FILTER"

d = plt.imread("def_red_blood.png")[:,:,0]

#PSF obtained from a circle of radius R=0.03:

x = np.linspace(-1, 1, 512)
X, Y = np.meshgrid(x, x)
d1 = np.sqrt(X * X + Y * Y)
circ = d1 < 0.03

#Ahora hago la transformada de fourier de todo esto para obtener el operador que
#me convertirá la imagen enfocada en una desenfocada con el producto de convolución
psf= np.abs(np.fft.fftshift(np.fft.fft2(circ)))
#Tiene que ir al cuadrado
psf = psf**2
#Observo que el filtro no está normalizado, lo normalizo y enchufo todo como uint8
psf = 255*(psf/np.max(psf))



D = np.fft.fft2(d)
H = np.fft.fft2(psf)
k = 1000000

d_r = np.abs(np.fft.ifft2( D*(1./(H+k))  ))

plt.figure()

plt.subplot(121)
plt.imshow(d,cmap='gray')
plt.axis('off')

plt.subplot(122)
plt.imshow(d_r,CMAP='gray')
plt.title('Inverse filter reconstruction')
plt.axis('off')


#%% 

"Weiner filter"

k = 15000000000
d_weiner = np.abs(np.fft.ifftshift(np.fft.ifft2( D*(np.conj(H)/(np.abs(H)**2+k)) ) ))

plt.figure()

plt.subplot(121)
plt.title('Original Defocused Image')
plt.imshow(d,cmap='gray')
plt.axis('off')

plt.subplot(122)
plt.title('Weiner filter reconstruction')
plt.axis('off')

plt.imshow(d_weiner,CMAP='gray')


#%% 

"Least Squares filter"

u, v = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))

k = 10e9
d_lsf = np.abs(np.fft.ifftshift(np.fft.ifft2( D*(np.conj(H)/(np.abs(H)**2+k*(u**2+v**2))) )) )

plt.figure()

plt.subplot(121)
plt.title('Original Defocused Image')
plt.imshow(d,cmap='gray')
plt.axis('off')

plt.subplot(122)
plt.title('Least squares filter reconstruction')
plt.axis('off')
plt.imshow(d_lsf,cmap='gray')






