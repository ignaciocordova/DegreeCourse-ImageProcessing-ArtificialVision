#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:07:30 2022

Lab # 5: Fourier transforms and spatial filtering.
5.1 Basic operations.
5.2 Fourier series and filtering of spatial frequencies.
5.3. Relative importance of amplitude and phase of the Fourier Transform. 
5.4. Spatial filtering.
    5.4.1. Sharp cut-off low-pass filter. 
    5.4.2. Laplacian filter.
    5.4.3. Gaussian filter.
    5.4.4. Butterworth filters.
    5.4.5. Quasi-periodic noise filtering. 
5.5. Spatial filtering in the image domain.
    5.5.1. Linear convolution kernels.
    5.5.2. The Kirsch compass kernel. 
    5.5.3. Salt and Pepper noise.
    5.5.4. Roberts, Sobel and Prewitt filters.


@author: Ignacio Cordova
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
from scipy import ndimage
from skimage.util import random_noise
import skimage.data as data
import scipy.ndimage.filters as filt
import numpy.random as random
from scipy import signal
from scipy.signal import chirp, spectrogram
from skimage.transform import resize


"""5.1 Basic operations.

Starting from an 8-bit test image, perform the following:
    
1. Calculate the 2D Fast Fourier Transform (FFT) using numpy.fft.fft2() and
      numpy.fft.fftshift().
      
2. Display the amplitude and phase of the FT. Use numpy.abs() and 
numpy.angle() (real and imaginary part are not interesting. Why?)

3. Inverse Fourier transform the result (use numpy.fft.ifft2() and
numpy.fft.ifftshift()). Try to recover the original signal.

4. Note that the amplitude can take very high values at the center. 
Calculate the base-10logarithm of the amplitude of the Fourier transform. 
Alternatively saturate the values over a certain arbitrary threshold 
(e.g. 0.1%, or 0.01% of the maximum value of the amplitude of the 
 Fourier transform). Display the result.
"""

#Para hacerlo cojo una imagen en blanco y negro,
#si es en color habrá que hacerlo para cada canal


im = data.astronaut()[:,:,0]

#transformada de Fourier type= COMPLEX the "2" in fft2 indicates dim2
tf = np.fft.fftshift(np.fft.fft2(im))

#amplitude of complex number
amp = np.log10(np.abs(tf))
#phase of complex number
phi = np.angle(tf)

#inverse fourier transform indicates complex variable
#np.fft.ifft2(np.fft.ifftshift(tf)) is complex, we take amplitude
recovered = np.abs(np.fft.ifft2(np.fft.ifftshift(tf)))


plt.figure()

plt.subplot(221)
plt.title('FFT amplitude')
plt.axis('off')
plt.imshow(amp)

plt.subplot(222)
plt.title('FFT phase')
plt.axis('off')
plt.imshow(phi)

plt.subplot(223)
plt.title('Original Image')
plt.axis('off')
plt.imshow(im,cmap='gray')

plt.subplot(224)
plt.title('Recovered Image')
plt.axis('off')
plt.imshow(recovered,cmap='gray')



"""5.2 Fourier series and filtering of spatial frequencies

Using function scipy.signal.square, generate a 5Hz square wave with a sampling 
frequency of 500 samples per second (see the example in the documentation of the 
function). Calculate the 1D-TF (with numpy.fft.fft()and numpy.fft.fftshift()) 
and filter the high frequency harmonics except those that correspond to a 
sinusoidal signal. Compute the inverse TF and show the result.
"""


#A 5 Hz waveform with 500 points in 1 second:
t = np.linspace(0, 1, 500, endpoint=False)
signal = signal.square(2 * np.pi * 5 * t)

sin = np.sin(2*np.pi*5*t)

signal3 = chirp(t, f0=50, f1=1, t1=1, method='linear')


# 1-dimensional Fourier Transform 
tf_signal = np.fft.fftshift(np.fft.fft(signal))

#ens quedem amb l'amplitud
tf = abs(tf_signal)

tf2 = abs(np.fft.fftshift(np.fft.fft(sin)))

tf3 = abs(np.fft.fftshift(np.fft.fft(signal3)))

plt.figure()


plt.subplot(321)
plt.title('Signals')
plt.plot(signal)
plt.ylim(-2, 2)

plt.subplot(322)
plt.title('Fourier Transforms')
plt.plot(tf)

plt.subplot(323)
plt.plot(sin)

plt.subplot(324)
plt.plot(tf2)

plt.subplot(325)
plt.plot(signal3)

plt.subplot(326)
plt.plot(tf3)

#%%
"""Filttering en 2D PREGUNTA!! El filtro (mask) lo aplico solo a la amplitud a la
Transformada de Fourier con todo??? """

l = 1 #longitud
freq = 5
n = 500 #nº of samples (pixels)

#Distancia entre pixels:
t = l/n
fn = 1/2*t #cut-off frequency

#Las frecuencias accesibles irán de -fn a fn. Genero un 
# "código de barras" con meshgrid (señal 2D)

x,y = np.meshgrid(np.linspace(0, l, n), np.linspace(0, l, n))
bars = sp.signal.square(2 * np.pi * freq * x)

#Hago la transformada de Fourier (OJO! AHORA ES EN 2D, 
#HAY QUE PONER EL 2 EN FFT)
tf_bars = np.fft.fftshift(np.fft.fft2(bars))




#Para el filtro
filtre = np.ones([500, 500])
filtre[230:270, 230:270] = 0


"""PREGUNTA!! El filtro (mask) lo aplico solo a la amplitud a la
Transformada de Fourier con todo??? """
bars_recovered = np.abs(np.fft.ifft2(np.fft.ifftshift(filtre*tf_bars)))

filtre2 = np.zeros([500, 500])
filtre2[240:260, 240:260] = 1

"""PREGUNTA!! El filtro (mask) lo aplico solo a la amplitud a la
Transformada de Fourier con todo??? """
bars_recovered2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtre2*tf_bars)))


plt.figure()

plt.subplot(321)
plt.title('Original Image')
plt.axis('off')
plt.imshow(bars,cmap='gray')

plt.subplot(322)
plt.title('Fourier Transform')
plt.axis('off')
plt.imshow(np.abs(tf_bars),cmap='gray')

plt.subplot(323)
plt.title('Pasa altos, keeps high frequencies, loses main structure')
plt.axis('off')
plt.imshow(filtre,cmap='gray')

plt.subplot(324)
plt.title('Inverse FT ')
plt.axis('off')
plt.imshow(bars_recovered,cmap='gray')

plt.subplot(325)
plt.title('Pasa bajos, keeps low frequencies, loses detail ')
plt.axis('off')
plt.imshow(filtre2,cmap='gray')

plt.subplot(326)
plt.title('Inverse FT')
plt.axis('off')
plt.imshow(bars_recovered2,cmap='gray')


#%%

""" Otro filtering en 2D más interesante"""


#Para el filtro
filtre = np.ones([500, 500])
filtre[230:270, 230:270] = 0

filtre2 = np.zeros([500, 500])
filtre2[240:260, 240:260] = 1


im = resize(plt.imread('image.png')[:,:,0],(500,500))
tf_im = np.fft.fftshift(np.fft.fft2(im))

abs_tf_im = np.abs(tf_im)



im_recovered = np.abs(np.fft.ifft2(np.fft.ifftshift(filtre*tf_im))) 
im_recovered2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtre2*tf_im))) 



plt.figure()

plt.subplot(321)
plt.imshow(im,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(322)
plt.title('FT AMPLITUDE')
plt.imshow(np.log10(np.abs(tf_im)),cmap='gray')
plt.axis('off')

plt.subplot(323)
plt.title('Pasa altos, keeps high frequencies, loses main structure')
plt.axis('off')
plt.imshow(filtre,cmap='gray')

plt.subplot(324)
plt.title('Inverse FT ')
plt.axis('off')
plt.imshow(im_recovered,cmap='gray')

plt.subplot(325)
plt.title('Pasa bajos, keeps low frequencies, loses detail ')
plt.axis('off')
plt.imshow(filtre2,cmap='gray')

plt.subplot(326)
plt.title('Inverse FT')
plt.axis('off')
plt.imshow(im_recovered2,cmap='gray')



#%%
"""
5.3. Relative importance of amplitude and phase of the
 Fourier Transform. 
 """


im1 = data.astronaut()[:,:,1]
im2 = data.camera()
#1) Calculo su transformada de fourier

im1_tf = np.fft.fftshift(np.fft.fft2(im1))
im2_tf = np.fft.fftshift(np.fft.fft2(im2))

#calculo su amplitud
abs_im1_tf = np.abs(im1_tf)
abs_im2_tf = np.abs(im2_tf)

#y su fase 
fase1 = np.angle(im1_tf)
fase2 = np.angle(im2_tf)

#Intercambio sus amplitudes y sus fases
#La transformada es la amplitud por una exponencial:
swap1 = abs_im1_tf*np.exp(1j*fase2)
swap2 = abs_im2_tf*np.exp(1j*fase1)

#Hago la transformada inversa

inv_swap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(swap1)))
inv_swap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(swap2)))


#Ahora calculo las fases de las imágenes originales y 
#hago la FT inversa solo conla fase con amplitud = 1
fase1_only = 1*np.exp(1j*fase1)
fase2_only = 1*np.exp(1j*fase2)

inv_fase1_only= np.abs(np.fft.ifft2(np.fft.ifftshift(fase1_only)))
inv_fase2_only= np.abs(np.fft.ifft2(np.fft.ifftshift(fase2_only)))

plt.figure()
plt.subplot(2,3,1, title= 'Original Astronauta')
plt.imshow(im1, cmap ='gray')
plt.subplot(2,3,3, title = 'Solo fase Astronauta (tiene mucha info!) ')
plt.imshow(inv_fase1_only, cmap='gray')
plt.subplot(2,3,2, title= 'Original Cameraman')
plt.imshow(im2, cmap ='gray')
plt.subplot(2,3,4, title= 'Solo fase cameraman (tiene mucha info!)')
plt.imshow(inv_fase2_only, cmap= 'gray')
plt.subplot(2,3,5, title= 'Im1 fase 2')
plt.imshow(inv_swap1, cmap= 'gray')
plt.subplot(2,3,6, title= 'Im2 fase 1')
plt.imshow(inv_swap2, cmap= 'gray')

#%%

"""
5.4. Spatial filtering.
    5.4.1. Sharp cut-off low-pass filter. 
    5.4.2. Laplacian filter.
    5.4.3. Gaussian filter.
    5.4.4. Butterworth filters.
    5.4.5. Quasi-periodic noise filtering.
    
    
    5.4.1. Sharp cut-off low-pass filter.
"""

im = data.camera()

F = np.fft.fftshift(np.fft.fft2(im))

    
NP = 512
U, V = np.meshgrid(np.linspace(-1, 1, NP), np.linspace(-1, 1, NP))


radius = 0.1 # e.g.
d = np.sqrt(U * U + V * V)
Lp = d < radius


A = 50
Gf = np.exp(-A*(U * U + V * V))# set A

Lf = U * U + V * V

C1 = np.fft.ifft2(np.fft.ifftshift(F * Lp))
C2 = np.fft.ifft2(np.fft.ifftshift(F * Gf))
C3 = np.fft.ifft2(np.fft.ifftshift(F * Lf))

plt.figure(tight_layout=True,figsize=(9,9))

plt.subplot(422)
plt.imshow(im,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(421)
plt.title('FT AMPLITUDE')
plt.imshow(np.log10(np.abs(F)),cmap='gray')
plt.axis('off')

plt.subplot(423)
plt.title('Sharp cut-off low-pass')
plt.axis('off')
plt.imshow(Lp,cmap='gray')

plt.subplot(424)
plt.title('Inverse FT ')
plt.axis('off')
plt.imshow(np.abs(C1),cmap='gray')

plt.subplot(425)
plt.title('Gaussian filter ')
plt.axis('off')
plt.imshow(Gf,cmap='gray')

plt.subplot(426)
plt.title('Inverse FT')
plt.axis('off')
plt.imshow(np.abs(C2),cmap='gray')

plt.subplot(427)
plt.title('Laplacian filter')
plt.axis('off')
plt.imshow(Lf,cmap='gray')

plt.subplot(428)
plt.title('Inverse FT')
plt.axis('off')
plt.imshow(np.abs(C3),cmap='gray')

#%% 

im = resize(plt.imread('clown.jpeg')[:,:,0],(500,500))
tf_im = np.fft.fftshift(np.fft.fft2(im))
abs_tf_im = np.abs(tf_im)


filtre = np.ones([500,500])
filtre[220:240, 290:310] = 0
filtre[270:290, 200:220] = 0
filtre[258:265, 265:280] = 0
filtre[230:245, 220:240] = 0


im_recovered = np.abs(np.fft.ifft2(np.fft.ifftshift(filtre*tf_im))) 




plt.figure()

plt.subplot(221)
plt.title('Original Image')
plt.axis('off')
plt.imshow(im,cmap='gray')

plt.subplot(222)
plt.title('TF Amplitude')
plt.imshow(np.log10(abs_tf_im))

plt.subplot(223)
plt.title('Filter')
plt.imshow(filtre,cmap='gray')

plt.subplot(224)
plt.title('Recovered Image')
plt.imshow(np.abs(im_recovered),cmap='gray')


#%% 

"""5.5. Spatial filtering in the image domain.
    5.5.1. Linear convolution kernels.
    5.5.2. The Kirsch compass kernel. 
    5.5.3. Salt and Pepper noise.
    5.5.4. Roberts, Sobel and Prewitt filters.
"""
im = np.double(data.camera())

#Laplacian-like (edge extraction) filters
weights1 = [[0,-1,0],
           [-1,4,-1],
           [0,-1,0]]

weights2 = [[-1,-1,-1],
           [-1,8,-1],
           [-1,-1,-1]]

weights3 = [[1,-2,1],
           [-2,4,-2],
           [1,-2,1]]


#els resultats sortiran amb valors negatius també!! El que ens
# interessa és el mòdul!!! 
result1 = sp.ndimage.filters.convolve(im, weights1, 
                                        output=None, 
                                        mode='reflect', 
                                        cval=0.0, 
                                        origin=0)

result2 = sp.ndimage.filters.convolve(im, weights2, 
                                        output=None, 
                                        mode='reflect', 
                                        cval=0.0, 
                                        origin=0)

result3 = sp.ndimage.filters.convolve(im, weights3, 
                                        output=None, 
                                        mode='reflect', 
                                        cval=0.0, 
                                        origin=0)

plt.figure()

plt.subplot(221)
plt.imshow(im,cmap='gray')

plt.subplot(222)
plt.imshow(np.abs(result1),cmap='gray')

plt.subplot(223)
plt.imshow(np.abs(result2),cmap='gray')

plt.subplot(224)
plt.imshow(np.abs(result3),cmap='gray')


#%%

filt = np.array( [[0., -1., 0.],
                  [-1., 4., -1.], 
                  [0., -1., 0.]])
 
filt512 = np.zeros([512, 512]) 
filt512[255:258, 255:258] = filt
ftfilt = np.fft.fftshift(np.fft.fft2(filt512))

u, v = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))
lap = u**2 + v**2

ftim = np.fft.fftshift(np.fft.fft2(im))

#Un producte a l'espai de Fourier es el mateix que una
# convolució a l'espai real
res = np.abs(np.fft.ifft2(np.fft.ifftshift(ftim * lap)))
conv = np.abs(ndimage.convolve(im, filt))


plt.figure()
plt.subplot(1,2,1)
plt.imshow(res, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(conv, cmap='gray')

tflap = np.real(np.fft.ifftshift(np.fft.fft2(lap)))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(tflap, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(tflap[255:258, 255:258], cmap='gray')
print(4 * tflap[255:258, 255:258] / tflap.max())


#%%

#5.5.2 KIRSCH COMPASS KERNEL
#Este algoritmo detecta bordes(?) en direcciones predeterminadas. La magnitud borde
# del operador de Kirsch se calcula con cada píxel como la máxima magnitud a través
#de todas las direcciones

im2d = np.double(data.cat()[:,:,1])

#Es una matriz compuesta de 8 matrices 3x3
kir = np.zeros([8, 3, 3])
g1 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
g2 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
g3 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
g4 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
g5 = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
g6 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
g7 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
g8 = np.array([[-3, 5, 5],[-3,0,5],[-3,-3,-3]])

kir[0,:,:] = g1
kir[1,:,:] = g2
kir[2,:,:] = g3
kir[3,:,:] = g4
kir[4,:,:] = g5
kir[5,:,:] = g6
kir[6,:,:] = g7
kir[7,:,:] = g8
#Lo convoluciono todo
#La matriz enorme en la que voy a meter las imágenes convolucionadas, cada una con su filtro:
#(Cada imagen estará convolucionada )
convo_kirsch=np.zeros((8, im2d.shape[0], im2d.shape[1]))

convo_kirsch[0,:,:] = np.abs(ndimage.convolve(im2d, g1))
convo_kirsch[1,:,:] = np.abs(ndimage.convolve(im2d, g2))
convo_kirsch[2,:,:] = np.abs(ndimage.convolve(im2d, g3))
convo_kirsch[3,:,:] = np.abs(ndimage.convolve(im2d, g4))
convo_kirsch[4,:,:] = np.abs(ndimage.convolve(im2d, g5))
convo_kirsch[5,:,:] = np.abs(ndimage.convolve(im2d, g6))
convo_kirsch[6,:,:] = np.abs(ndimage.convolve(im2d, g7))
convo_kirsch[7,:,:] = np.abs(ndimage.convolve(im2d, g8))

im_kirsch = np.zeros((im2d.shape[0],im2d.shape[1]))

for ii in range(im2d.shape[0]):
    for jj in range(im2d.shape[1]):
        im_kirsch[ii, jj] = np.amax(convo_kirsch[0:8, ii, jj])
                
plt.figure()
plt.imshow(im_kirsch, cmap='gray')

#%%
#5.5.3. SALT AND PEPPER NOISE
saltpepper = np.double(plt.imread('salt_pepper.png'))

#Le aplico el filtro
kernel7 = 1/9.*(np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]], 
                         dtype=np.double))

saltpepper_filtrada = np.abs(ndimage.convolve(saltpepper, kernel7))

#Ahora aplico el medfilt
saltpepper_medfilt = sp.signal.medfilt(saltpepper, kernel_size=7)


plt.figure()
plt.subplot(2,3,1)
plt.imshow(saltpepper)
plt.subplot(2,3,2)
plt.imshow(saltpepper_filtrada)
plt.subplot(2,3,3)
plt.imshow(saltpepper_medfilt)


#%%
#5.5.4 ROBERTS, SOBEL AND PREWITT FILTERS
robx = np.array([[1, 0], [0, -1]], dtype=np.double)
roby = np.array([[0, 1], [-1, 0]], dtype=np.double)
sobx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.double)
soby = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.double)
prex = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.double)
prey = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.double)

roberts = np.sqrt(ndimage.convolve(im2d, robx)**2 + ndimage.convolve(im2d, roby)**2)
sobels = np.sqrt(ndimage.convolve(im2d, sobx)**2 + ndimage.convolve(im2d, soby)**2)
prewitt = np.sqrt(ndimage.convolve(im2d, prex)**2 + ndimage.convolve(im2d, prey)**2)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(im2d, cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(roberts, cmap='gray')
plt.subplot(2, 2, 3)
plt.imshow(sobels, cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(prewitt, cmap='gray')


#%% 
# 5.6 Image alignment by cross-correlation
from numpy import unravel_index

boli0 = plt.imread('boli0.jpeg')[:,:,0]
boli1 = plt.imread('boli1.jpeg')[:,:,0]


corr = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(boli0)) *
np.conjugate(np.fft.fftshift(np.fft.fft2(boli1))))))


i,j = unravel_index(corr.argmax(), corr.shape)

plt.imshow(corr,cmap='gray')
plt.axvline(j)
plt.axhline(i)


boli_shifted = ndimage.shift(boli1, [-i+boli0.shape[0]/2.,-j+boli0.shape[1]/2])
