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
from mpl_toolkits.mplot3d import Axes3D

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
tf = np.fft.fftshift(np.fft.fft2(im))
mod_tf = np.abs(tf) #Haré el imshow de esto porque solo me interesa el módulo, no valores Re o Im
log_mod_tf = np.log10(mod_tf)

#plt.figure()


def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    fig.colorbar(surf)

    ax.set_xlabel('X (cols)')
    ax.set_ylabel('Y (rows)')
    ax.set_zlabel('Z (values)')

    plt.show()
    
    return

# m.shape must be (10,10)
#m = np.fromfunction(lambda x, y: np.sin(np.sqrt(x**2 + y**2)), (10, 10))

surface_plot(im, cmap=plt.cm.coolwarm)

