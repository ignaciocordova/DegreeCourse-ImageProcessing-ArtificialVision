#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:26:37 2022

Lab # 4: More on color and channel transformations.
4.1. RGB coordinates from spectrum data. The CIE 1931 XYZ color model. 
4.2. Histogram equalization.
4.3. Image entropy.
4.4. Least-significant bits steganography.
4.5. Visual encryption.


@author: Ignacio Cordova 
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk,square
from skimage.measure import shannon_entropy
from scipy.ndimage.measurements import histogram
import skimage.transform as st

import cordova_functions as cordova


#%%


"4.1. RGB coordinates from spectrum data. The CIE 1931 XYZ color model. "

#Tengo que calcular las integrales de un espectro de datos E(lambda), con las
#color matching functions x(lambda), y(), z(), y haciendo las integrales con el simpy

array = np.load('color-matching-functions_plus_spectrum.npy')

#Para hacer la integral tengo que usar scipy
#Para seleccionar las diferentes matrices
#VIGILA QUÉ matriz ES CADA COSA!!!!!!!!!!

lamb=np.copy(array[:,0])
x = np.copy(array[:,1])
y = np.copy(array[:,2])
z = np.copy(array[:,3])
E = np.copy(array[:,4])

plt.figure()
plt.subplot(121)
plt.plot(lamb,E,label='E(lambda)')
plt.ylim(0,1.80)
plt.subplot(122)
plt.plot(lamb,x,label='x',c='r')
plt.plot(lamb,y,label='y',c='g')
plt.plot(lamb,z,label='z',c='b')
plt.xlabel('Lambda')
plt.legend()

integralx = scipy.integrate.simpson(x*E, lamb)
integraly = scipy.integrate.simpson(y*E, lamb)
integralz = scipy.integrate.simpson(z*E, lamb)

#Normalizamos
X = integralx/(integralx+integraly+integralz)
Y = integraly/(integralx+integraly+integralz)
Z = integralz/(integralx+integraly+integralz)


#Tengo que tener un número para R, otro para G y otro para B
#Tengo que hacer la multiplicación matricial que sale en el enunciado

XYZ = np.array([X, Y, Z])
CIE = np.array([[3.240479, -1.537150, -0.498535],
                  [-0.969256, 1.875992, 0.041556],
                  [0.055648, -0.204043, 1.057311]])
rgb_1 = 255*(np.dot(CIE, XYZ))
rgb = np.uint8(rgb_1)
print(rgb)


#lets build a RGB image

imRGB = np.ones([250,250,3],dtype=np.uint8)


imRGB[:,:,0] = imRGB[:,:,0]*rgb[0]
imRGB[:,:,1] = imRGB[:,:,1]*rgb[1]
imRGB[:,:,2] = imRGB[:,:,2]*rgb[2]

plt.figure()
plt.imshow(imRGB)


#%%

"""4.2. Histogram equalization.

obtenim una imatge amb més contrast, de manera que les zones
 que semblen més uniformes a l’original, queden més degradades. 
 Per exemple, una imatge de poc contrast pot tenir els valors 
 repartits entre 50 i 100, després d’equalitzar-ne l’histograma, 
 ho estan entre 0 i 255. Un cas concret on això té molta utilitat
 són les imatges més fosques, on podem obtenir zones més clares."""
 


#Ahora veo si funciona con una imagen cualquiera (blanco y negro se verá más)
im = data.moon()
imeq = cordova.equalize(im)


#%%


"""4.3. Image entropy.
    S provides an idea of the theoretical compression limit.
    The local entropy can be calculated using 
    skimage.filters.rank.entropy
    For color images, the entropy should be calculated for every
    channel.
    
    Using the green channel of skimage.data.astronaut,
a) Determine the global entropy of the image.
b) Calculate the local entropy image using an 11x11 pixel neighborhood. 
You might want to display the result in combination of a colorbar().
"""

#a) Determine the global entropy of the image 

im = data.astronaut()[:,:,1]

#Calculo el histograma y lo normalizo (será la probabilidad)
prob = histogram(im, 0, 255, 256)/(512.*512.)

#entropia global
#añadimos un valor muy pequeño por si aparece prob=0 para que 
#el log no pete
entropia_global = -np.sum(prob*np.log2(prob+(1e-10))) 
print(entropia_global,'a pelo')

sh = shannon_entropy(im)
print(sh,'funcion shannon_entropy(im)')


#b) Calculate the local entropy image using an 11x11 pixel neighborhood. 
#You might want to display the result in combination of a colorbar().

ent = entropy(im,np.ones([11,11],dtype=np.uint8))

plt.figure()
plt.title('Entropy')
plt.imshow(ent)
plt.colorbar()



#%%

"""4.4. Least-significant bits steganography.
"""


secretim = np.uint8( 
    st.resize( plt.imread('satellite.jpg')[:,:,0], (512,512) ) *255 ) 

hostim = data.camera()



secret_bin = np.unpackbits(secretim).reshape(512,512,8)
host_bin = np.unpackbits(hostim).reshape(512,512,8)


#agafem els bits és significatius de la imatge host
#en els mnys significatius esrivim els més significatius de secret 


combined = 128 * host_bin[:,:,0] + \
            64 * host_bin[:,:,1] + \
            32 * host_bin[:,:,2] +\
            16 * host_bin[:,:,3] + \
            8 * host_bin[:,:,4] + \
            4 * host_bin[:,:,5] + \
            2 * secret_bin[:,:,0] + \
            secret_bin[:,:,1]
            
combined_bin = np.unpackbits(combined).reshape(512,512,8)


host_recuperada = 128 * combined_bin[:,:,0] + \
            64 * combined_bin[:,:,1] + \
            32 * combined_bin[:,:,2] +\
            16 * combined_bin[:,:,3] + \
            8 * combined_bin[:,:,4] + \
            4 * combined_bin[:,:,5]  

secret_recuperada = 128*combined_bin[:,:,6] + \
                    64*combined_bin[:,:,7]

            
plt.figure()
plt.figure(figsize = (12, 24))



plt.subplot(321)
plt.title('Host Image')
plt.imshow(data.camera(),cmap='gray')
plt.axis('off')

plt.subplot(322)
plt.title('Secret Image to hide')
plt.imshow(secretim,cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.subplot(323)
plt.title('Secret Image in Host Image')
plt.imshow(combined,cmap='gray')
plt.axis('off')


plt.subplot(325)
plt.title('6bit Host Image')
plt.imshow(host_recuperada,cmap='gray')
plt.axis('off')

plt.subplot(326)
plt.title('2bit Secret Image')
plt.imshow(secret_recuperada,cmap='gray')
plt.axis('off')

#%%