#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 11:57:22 2022

FUNCIONES ÚTILES PARA Image Processing and Artificial Vision

This is an earlier version of the IMPAVI module which you 
can find at https://github.com/ignaciocordova 

@author: Ignacio Cordova 
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage
from scipy.ndimage.measurements import histogram
from sklearn.cluster import KMeans
from scipy import ndimage


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


def channels(rgb):
    """
    Separates the channels of an RGB image.

    Input: 3-channel RGB image
    Output: 3 gray-scale images
    """
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    return r,g,b



#function that creates an image of with a circle of radius r centered in the image
def circle(r,n):
    """
    Creates a circle of radius r centered in the image.
    
    Inputs: radius n
    Output: circle image
    """
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    lf = u**2 + v**2
    circ = lf<r
    return circ



def equalize(im,plot=False):
    
    """"
    Equalize the histogram of a one-channel image.

    Inputs: gray-scale image and plot flag
    Output: equalized image
    """
    
    h = histogram(im,0,255,256)
    hc = np.cumsum(h)/(im.shape[0]*im.shape[1])
    
    im_eq = hc[im]
   
    h_im_eq = histogram(im_eq,0,1,256)
    h_im_eq_c = np.cumsum(h_im_eq)/(im.shape[0]*im.shape[1])

    if plot:

        plt.figure(constrained_layout=True,figsize=(20,20))

        plt.subplot(321)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')


        plt.subplot(322)    
        plt.plot(h)
        plt.title('Original Histogram')

        plt.subplot(323)    
        plt.plot(255*hc)
        plt.title('Histogram Cumulative Sum Normalized of Original Image')

        plt.subplot(326)    
        plt.plot(h_im_eq)
        plt.title('Histogram of Equalized Image')

        plt.subplot(325)
        plt.imshow(im_eq,cmap='gray')
        plt.title('Equalized')
        plt.axis('off')


        plt.subplot(324)    
        plt.plot(255*h_im_eq_c)
        plt.title('Histogram Cumulative sum of Equalized Image')
        
    return im_eq


def high_pass_filter(im,radius,plot=False):
    """
    Applies a high pass filter to an image.

    Inputs: gray-scale image and radius of the filter and plot flag
    Output: high pass filtered image

    """
    im_tf = np.fft.fftshift(np.fft.fft2(im))
    #build Laplacian Filter 
    u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
    lf = u**2 + v**2
    #sharp cut-off
    circ = lf>radius

    im1 = np.abs(np.fft.ifft2(im_tf*circ))

    if plot:
        plt.figure(constrained_layout=True,figsize=(20,20))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(im1,cmap='gray')
        plt.title('Laplacian Filter')
        plt.axis('off')

    return im1

def low_pass_filter(im,radius,plot=False):
    """
    Applies a low pass filter to an image.
    
    Inputs: gray-scale image and radius of the filter and plot flag
    Output: low pass filtered image
    """
    im_tf = np.fft.fftshift(np.fft.fft2(im))
    #build Laplacian Filter 
    u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
    lf = u**2 + v**2
    #sharp cut-off
    circ = lf<radius

    im1 = np.abs(np.fft.ifft2(im_tf*circ))

    if plot:
        plt.figure(constrained_layout=True,figsize=(7,7))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(im1,cmap='gray')
        plt.title('Laplacian Filter')
        plt.axis('off')

    return im1


def median_filters(im,plot=False):
    """
    Applies a series of median filters to an image.

    Inputs: gray-scale image and plot flag
    Output: median filtered image
    """

    media1 = scipy.signal.medfilt2d(im, kernel_size=[11,11])
    media2 = scipy.signal.medfilt2d(media1, kernel_size=[41,41])
    media3 = scipy.signal.medfilt2d(media2, kernel_size=[21,21])
    media4 = scipy.signal.medfilt2d(media3, kernel_size=[21,21])
    if plot:
        plt.figure(constrained_layout=True,figsize=(20,20))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(media2,cmap='gray')
        plt.title('Double Median Filter')
        plt.axis('off')
    return media2


def FS(ast):
       
    """
    Floyd-Steinbeirg (FS) algorithm for error compensation 

    Input: ast- 2D array (gray scale) normalized to 1
    Output: Binarized image 
    """
    
    
    for i in range(1,511): #skips the edges 
        for j in range(1,511):
            
            #error 
            px = ast[i,j]
            if px>0.5:
                error = px-1

            else:
                error = px
            
            ast[i,j+1]=ast[i,j+1]     + (7./16.)*error
            ast[i+1,j+1]=ast[i+1,j+1] + (1./16.)*error
            ast[i+1,j]=ast[i+1,j]     + (5./16.)*error
            ast[i+1,j-1]=ast[i+1,j-1] + (3./16.)*error
                
                
    imFS = ast>0.5
    return imFS 


def otsu_filter(im,plot=False):
    """
    Applies an Otsu threshold to an image.
    
    Inputs: gray-scale image and plot flag
    Output: Otsu filtered image
    """
    otsu = im>skimage.filters.threshold_otsu(im)
    if plot:
        plt.figure(constrained_layout=True,figsize=(7,7))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(otsu,cmap='gray')
        plt.title('Otsu Filter')
        plt.axis('off')
    return otsu



def JJN(ast):
    """
    Jarvis-Judice-Ninke (JJN) algorithm for error compensation
    
    Input: ast- 2D array (gray scale) normalized to 1
    Output: Binarized image
    """

    for ii in range(1,510): #skips the edges
        for jj in range(1,510):
            
            #error
            px = ast[ii,jj]
            if px>0.5:
                error = px-1
                
            else:
                error = px
            
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
    
    
    imJJN = ast>0.5
    return imJJN 



def kirsch_compass_kernel(im):
    """
    Applies a Kirsch compass filter to an image.
    
    Inputs: gray-scale image
    Output: Kirsch compass filtered image
    """

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
    
    convo_kirsch=np.zeros((8, im.shape[0], im.shape[1]))

    convo_kirsch[0,:,:] = np.abs(ndimage.convolve(im, g1))
    convo_kirsch[1,:,:] = np.abs(ndimage.convolve(im, g2))
    convo_kirsch[2,:,:] = np.abs(ndimage.convolve(im, g3))
    convo_kirsch[3,:,:] = np.abs(ndimage.convolve(im, g4))
    convo_kirsch[4,:,:] = np.abs(ndimage.convolve(im, g5))
    convo_kirsch[5,:,:] = np.abs(ndimage.convolve(im, g6))
    convo_kirsch[6,:,:] = np.abs(ndimage.convolve(im, g7))
    convo_kirsch[7,:,:] = np.abs(ndimage.convolve(im, g8))

    im_kirsch = np.zeros((im.shape[0],im.shape[1]))

    for ii in range(im.shape[0]):
        for jj in range(im.shape[1]):
            im_kirsch[ii, jj] = np.amax(convo_kirsch[0:8, ii, jj])
                    
    return im_kirsch


def color_reduction(im):
    """
    Performs color dithering followinf the HSV color model.
    RGB images have 256x256x256 possible color combinations
    This function reduces the colors to 6x6x6 possible combinations

    Inputs: RGB image
    Output: Dithered image"""
    
    
    #Build the look up table 
    array = np.arange(0,256,1)
    a = np.rint(array//43)*43
    
    
    #apply to RGB image 
    im6x6x6 = a[im]
    
    #Output is float so we MUST normalize! 
    im6x6x6 = im6x6x6/im6x6x6.max()
    
    return im6x6x6


"The next 2 functions are useful for applying K-means"

def reshape_channel(channel):
  "Reshapes a 2D array into a column vector"
  return channel.reshape(channel.shape[0]*channel.shape[1],1)

def reshape_rgb(rgb):
  "Reshapes an RGB image into a 4 column matrix"
  #impavi function separates the 3 channels
  r,g,b = impavi.channels(rgb)
  #arange them into one big matrix of size 3*(n*m)
  return np.concatenate([reshape_channel(r),
                          reshape_channel(g),
                          reshape_channel(b)], axis=1)


def apply_kmeans(im,k,plot=False):
    """
    Applies k-means clustering to an image and automatically selects the largest cluster.
    
    Inputs: gray-scale image and number of clusters and plot flag
    Output: largest cluster
    """
    dataK = im.reshape(im.shape[0]*im.shape[1],1)
    kmn = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(dataK)
    # a label (0 to k) is assigned to each sample (row)
    labels = kmn.predict(dataK)

    # from 1d-array to 2d-array
    imRes = np.reshape(labels, [im.shape[0], im.shape[1]])

    ref = 0 

    for label in range(k):
        cluster_size = np.count_nonzero(imRes==label)
        if cluster_size>ref:
            ref = cluster_size
            final_res = imRes==label
    
    if plot:
        plt.figure(constrained_layout=True,figsize=(11,11))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Input')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(imRes,cmap='gray')
        plt.title('Largest K-means cluster')
        plt.axis('off')
    
    return 1-final_res

