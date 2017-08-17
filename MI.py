# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 17:17:07 2017

@author: Administrator
"""
import __future__
import cv2
import numpy as np

def MI(a,b):
    Ma,Na = a.shape
    Mb,Nb = b.shape
    M = min(Ma,Mb)
    N = min(Na,Nb)

    hab = np.zeros((256,256))
    ha = np.zeros((1,256))
    hb = np.zeros((1,256))
    
    if np.amax(a) != np.amin(a):
        a = (a - np.amin(a))/float((np.amax(a)-np.amin(a)))
    else:
        a = np.zeros((M,N))
        
    if np.amax(b) != np.amin(b):
        b = (b - np.amin(b))/float((np.amax(b)-np.amin(b)))
    else:
        b = np.zeros((M,N))
      
    a = np.float64(np.int16(a*255))
    b = np.float64(np.int16(b*255))
    
    for i in range(M):
        for j in range(N):
            indexx = int(a[i,j])
            indexy = int(b[i,j])
            hab[indexx,indexy] = hab[indexx,indexy]+1
            ha[0,indexx] = ha[0,indexx]+1
            hb[0,indexy] = hb[0,indexy]+1

    hsum = np.sum(np.sum(hab))
    index0,index1 = np.where(hab !=0)
    p = hab/float(hsum)
    Hab = np.sum(np.sum(-p[index0,index1]*np.log(p[index0,index1]))) 
    
    hsum = np.sum(np.sum(ha))
    index0,index1 = np.where(ha !=0)
    p = ha/hsum
    Ha = np.sum(np.sum(-p[index0,index1]*np.log(p[index0,index1]))) 
    
    hsum = np.sum(np.sum(hb))
    index0,index1 = np.where(hb !=0)
    p = hb/float(hsum)
    Hb = np.sum(np.sum(-p[index0,index1]*np.log(p[index0,index1])))
    
    mi = Ha+Hb-Hab
    #mi = (Ha+Hb)/Hab
    return mi

if __name__ == '__main__':   
    img1 = cv2.imread("testdata/w1.jpg")
    img2 = cv2.imread("testdata/w2.jpg")
    
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    mi = MI(gray1,gray2)     
    