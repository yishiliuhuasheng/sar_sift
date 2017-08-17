# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:25:50 2017

@author: Administrator
"""
import __future__	
import numpy as np
import cv2
import scipy.ndimage
import math

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.float64(np.exp(-((x**2 + y**2)/(2.0*sigma**2))))
    return g/g.sum()
    
def build_scale(image, sigma, Mmax, ratio, d):
    M,N = image.shape
    sar_harris_function = np.zeros((M,N,Mmax))
    gradient = np.zeros((M, N, Mmax))
    angle = np.zeros((M, N, Mmax))
    
    for i in range(Mmax):
        scale = float(sigma*ratio**(i))
        radius = int(round(2*scale))
        j = list(range(-radius,radius+1,1))
        k = list(range(-radius,radius+1,1))
        xarry,yarry = np.meshgrid(j,k)
        W = np.exp(-(np.abs(xarry)+np.abs(yarry))/scale)
        W34 = np.zeros((2*radius+1,2*radius+1),dtype=float)
        W12 = np.zeros((2*radius+1,2*radius+1),dtype=float)
        W14 = np.zeros((2*radius+1,2*radius+1),dtype=float)
        W23 = np.zeros((2*radius+1,2*radius+1),dtype=float)
        
        W34[radius+1:2*radius+1,:] = W[radius+1:2*radius+1,:]
        W12[0:radius,:] = W[0:radius,:]
        W14[:,radius+1:2*radius+1] = W[:,radius+1:2*radius+1]
        W23[:,0:radius] = W[:,0:radius]

        M34 = scipy.ndimage.correlate(image, W34, mode='nearest')
        M12 = scipy.ndimage.correlate(image, W12, mode='nearest')
        M14 = scipy.ndimage.correlate(image, W14, mode='nearest')
        M23 = scipy.ndimage.correlate(image, W23, mode='nearest')
        
        Gx = np.log(M14/M23)
        Gy = np.log(M34/M12)
        
        Gx[np.where(np.imag(Gx))] = np.abs(Gx[np.where(np.imag(Gx))])
        Gy[np.where(np.imag(Gy))] = np.abs(Gy[np.where(np.imag(Gy))])
        Gx[np.where(np.isfinite(Gx)==0)] = 0
        Gy[np.where(np.isfinite(Gy)==0)] = 0
           
        gradient[:,:,i] = np.sqrt(np.square(Gx)+np.square(Gy))
        temp_angle = np.arctan2(Gy, Gx)
        
        temp_angle = temp_angle/math.pi*180
        temp_angle[np.where(temp_angle<0)] = temp_angle[np.where(temp_angle<0)]+360
        angle[:,:,i] = temp_angle
        
        Csh_11 = scale**2 * np.square(Gx)
        Csh_12 = scale**2 * Gx*Gy
        Csh_22 = scale**2 * np.square(Gy)
        
        gaussian_sigma = math.sqrt(2)*scale
        width = round(3*gaussian_sigma)
        width_windows = int(2*width+1)
        W_gaussian = fspecial_gauss(width_windows,gaussian_sigma)
        
        l = list(range(0,width_windows,1))
        m = list(range(0,width_windows,1))
        a,b = np.meshgrid(l,m)
        index0,index1 = np.where((np.square(a-width)-1)+np.square(b-width- 1)>width**2)
        W_gaussian[index0,index1] = 0
        
        Csh_11 = scipy.ndimage.correlate(Csh_11, W_gaussian, mode='nearest')
        Csh_12 = scipy.ndimage.correlate(Csh_12, W_gaussian, mode='nearest')
        Csh_21 = Csh_12
        Csh_22 = scipy.ndimage.correlate(Csh_22, W_gaussian, mode='nearest')
        
        sar_harris_function[:,:,i] = Csh_11*Csh_22-Csh_21*Csh_12-d*(Csh_11+Csh_22)**2
        
    return sar_harris_function,gradient,angle
    