"""
Created on May 17 2022

Module of useful functions for Image Processing and Artificial Vision
for the project of K-means clustering on Image Entropy and Edge enhanced
images to monitor cell migration in wound healing assays. 

@author: Ignacio Cordova 
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import skimage
from scipy.ndimage.measurements import histogram
from skimage.filters.rank import entropy
from sklearn.cluster import KMeans
from scipy import ndimage


def luminance(rgb):
    l_im =  0.299*np.double(rgb[:,:,0])+\
            0.587*np.double(rgb[:,:,1])+\
            0.114*np.double(rgb[:,:,2])

    return l_im 


def show(string):
    im = plt.imread(string)
    plt.figure(constrained_layout=True,figsize=(8,8))
    plt.imshow(im,cmap='gray')
    plt.axis('off')


def equalize(im,plot=False):
    
    """"
    
    """
    
    h = histogram(im,0,255,256)
    hc = np.cumsum(h)/(im.shape[0]*im.shape[1])
    
    im_eq = hc[im]
   
    h_im_eq = histogram(im_eq,0,1,256)
    h_im_eq_c = np.cumsum(h_im_eq)/(im.shape[0]*im.shape[1])

    if plot==True:

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


def laplacian_filter(im,radius,plot=False):
    im_tf = np.fft.fftshift(np.fft.fft2(im))
    #build Laplacian Filter 
    u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
    lf = u**2 + v**2
    circ = lf>radius

    im1 = np.abs(np.fft.ifft2(im_tf*circ))

    if plot==True:
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
    im_tf = np.fft.fftshift(np.fft.fft2(im))
    #build Laplacian Filter 
    u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
    lf = u**2 + v**2
    circ = lf<radius

    im1 = np.abs(np.fft.ifft2(im_tf*circ))

    if plot==True:
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
    media1 = scipy.signal.medfilt2d(im, kernel_size=[11,11])
    media2 = scipy.signal.medfilt2d(media1, kernel_size=[41,41])
    media3 = scipy.signal.medfilt2d(media2, kernel_size=[21,21])
    media4 = scipy.signal.medfilt2d(media3, kernel_size=[21,21])
    if plot==True:
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

def otsu_filter(im,plot=False):
    otsu = im>skimage.filters.threshold_otsu(im)
    if plot==True:
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

def kirsch_compass_kernel(im):

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

def apply_kmeans(im,k,plot=False):
    dataK = im.reshape(im.shape[0]*im.shape[1],1)
    kmn = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(dataK)
    # a label (0 to k) is assigned to each sample (row)
    labels = kmn.predict(dataK)

    centroids = kmn.cluster_centers_
    # from 1d-array to 2d-array
    imRes = np.reshape(labels, [im.shape[0], im.shape[1]])

    ref = 0 

    for label in range(k):
        cluster_size = np.count_nonzero(imRes==label)
        if cluster_size>ref:
            ref = cluster_size
            final_res = imRes==label
    
    if plot==True:
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


def hfreq_segmentation(string):
    im = plt.imread(string)
    im_eq = equalize(im,plot=False)
    im_eq_laplacian = laplacian_filter(im_eq,radius=0.005,plot=False)
    median = median_filters(im_eq_laplacian,plot=False)
    im_seg = apply_kmeans(median,7,plot=False)
    
    return im_seg

def edge_enhanced_segmentation(string):
    im = plt.imread(string)
    im_eq = equalize(im,plot=False)
    im_eq_kirsch = kirsch_compass_kernel(im_eq)
    median = median_filters(im_eq_kirsch,plot=False)
    im_seg = apply_kmeans(median,2,plot=False)
    
    return im_seg


def get_ssim(string,mode=0):
    im_manual = plt.imread(string+'_manual.png')
    im_topman = plt.imread(string+'_topman.png')
    im_tscratch = plt.imread(string+'_tscratch.png')
    im_multiceg_Seg = plt.imread(string+'_multiCellSeg.png')

    if mode == 0:
        my_segmentation = hfreq_segmentation(string+'.tif')
    if mode == 1: 
        my_segmentation = edge_enhanced_segmentation(string+'.tif')

    ssim_topman = skimage.metrics.structural_similarity(im_manual,im_topman)
    ssim_tscratch = skimage.metrics.structural_similarity(im_manual,im_tscratch)
    ssim_multiceg_Seg = skimage.metrics.structural_similarity(im_manual,im_multiceg_Seg)
    ssim_mine = skimage.metrics.structural_similarity(im_manual,my_segmentation)


    return [ssim_topman,ssim_tscratch,ssim_multiceg_Seg,ssim_mine]

def state_of_the_art(string):
    im_manual = plt.imread(string+'_manual.png')
    im_topman = plt.imread(string+'_topman.png')
    im_tscratch = plt.imread(string+'_tscratch.png')
    im_multiceg_Seg = plt.imread(string+'_multiCellSeg.png')

    ssim_topman = skimage.metrics.structural_similarity(im_manual,im_topman)
    ssim_tscratch = skimage.metrics.structural_similarity(im_manual,im_tscratch)
    ssim_multiceg_Seg = skimage.metrics.structural_similarity(im_manual,im_multiceg_Seg)
    plt.figure(figsize=(20,20))
    plt.subplot(141)
    plt.imshow(im_manual,cmap='gray')
    plt.title('Manually segmented (target)')

    plt.subplot(142)
    plt.imshow(im_topman,cmap='gray')
    plt.title('Topman SSIM={:.2f}'.format(ssim_topman))

    plt.subplot(143)
    plt.imshow(im_tscratch,cmap='gray')
    plt.title('Tscratch SSIM={:.2f}'.format(ssim_tscratch))

    plt.subplot(144)
    plt.imshow(im_multiceg_Seg,cmap='gray')
    plt.title('MultiCellSeg SSIM={:.2f}'.format(ssim_multiceg_Seg))


