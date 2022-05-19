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
import scipy.integrate
from skimage import data
from scipy.ndimage.measurements import histogram
import matplotlib
from scipy import signal
import scipy as scipy



def luminance(rgb):
    l_im =  0.299*np.double(rgb[:,:,0])+\
            0.587*np.double(rgb[:,:,1])+\
            0.114*np.double(rgb[:,:,2])

    return l_im 


def FS(ast):
       
    """3.2 Dithering
    
    Floyd-Steinbeirg (FS) algorithm for error compensation 
    
    INPUT: imagen array 2D (gray scale) NORMALIZED to 1 !!!  
    OUTPUT: binarized image """
    
    
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
    return imFS 


def JJN(ast):
    """3.2 Dithering
    
    Jarvis-Judice-Ninke (JJN) algorithm for error compensation 
    
    INPUT: imagen array 2D (gray scale) NORMALIZED to 1 !!!  
    OUTPUT: binarized image """
    
    

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
    
    
    imJJN = ast>0.5
    return imJJN 


def color_reduction(im):
    """3.3. Color dithering. The HSV color model.
    
    RGB images have 256x256x256 possible color combinations
    This function reduces the colors to 6x6x6 possible combinations
    255/43 = 5.93 """
    
    
    #Build the look up table 
    array = np.arange(0,256,1)
    a = np.rint(array//43)*43
    
    
    #apply to RGB image 
    im6x6x6 = a[im]
    
    #Output is float so we MUST normalize! 
    im6x6x6 = im6x6x6/im6x6x6.max()
    
    return im6x6x6

def equalize(im):
    
    """4.2. Histogram equalization.
    
    Calcula la imagen equalizada de un array 2D (gray scale) y plotea los 
    histogramas originales, ecualizados y cumulative sum
    
    INPUT: imagen array 2D 
    OUTPUT: imagen equalizada + plot completo """
    
    h = histogram(im,0,255,256)
    hc = np.cumsum(h)/(im.shape[0]*im.shape[1])
    
    im_eq = hc[im]
   
    h_im_eq = histogram(im_eq,0,1,256)
    h_im_eq_c = np.cumsum(h_im_eq)/(im.shape[0]*im.shape[1])

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