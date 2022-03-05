#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 19:25:41 2022

Other option for channel decomposition

@author: Ignacio Cordova Pou
"""

import matplotlib.pyplot as plt
import numpy as np 

#llegim la imatge
im = plt.imread('fresa.jpeg')

#dimensions i nº de capes de la imatge 
dim = im.shape

print("Alçada:",dim[0], "píxels")
print("Amplada:",dim[1], "píxels")
print("Capes:",dim[2])

#circle1 = plt.Circle((390, 280), 110, color='black',fill=False)
#circle2 = plt.Circle((390, 280), 110, color='black',fill=False)
#circle3 = plt.Circle((390, 280), 110, color='black',fill=False)

f, axarr = plt.subplots(2,3) # f is the figure, axarr is axes array


#les escales de grisos noms necessiten un array 2d

imr = im[:,:,0]
axarr[0,0].imshow(imr,cmap='gray')
axarr[0,0].set_title('Extracció R')
axarr[0,0].axis('off')
#axarr[0,0].add_patch(circle1)
#axarr[0,0].text(200, 150, 'Red car', fontsize=14)

img = im[:,:,1]
axarr[0,1].imshow(img,cmap = 'gray')
axarr[0,1].set_title('Extracció G')
axarr[0,1].axis('off')
#axarr[0,1].add_patch(circle2)

imb = im[:,:,2]
axarr[0,2].imshow(imb,cmap = 'gray')
axarr[0,2].set_title('Extracció B')
axarr[0,2].axis('off')
#axarr[0,2].add_patch(circle3)


#Ara anem a per els canals RGB 


imR = np.zeros((dim[0],dim[1],dim[2]),dtype=np.uint8)#creem 3 matrius de zeros
imR[:,:,0]=imr
axarr[1,0].imshow(imR)
axarr[1,0].set_title('R')
axarr[1,0].axis('off')

imG = np.zeros((dim[0],dim[1],dim[2]),dtype=np.uint8)#creem 3 matrius de zeros
imG[:,:,1]=img
axarr[1,1].imshow(imG)
axarr[1,1].set_title('G')
axarr[1,1].axis('off')

imB = np.zeros((dim[0],dim[1],dim[2]),dtype=np.uint8)#creem 3 matrius de zeros
imB[:,:,2]=imb
axarr[1,2].imshow(imB)
axarr[1,2].set_title('B')
axarr[1,2].axis('off')





