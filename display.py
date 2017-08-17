# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:32:56 2017

@author: Administrator
"""
import __future__
import cv2
import numpy as np
import random

color = [(255,0,0),(255,156,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255),(255,0,255)]
#display img1 img2,height must be same
def display(img1, img2, kp1, kp2):
    height1 = img1.shape[0]
    width1 = img1.shape[1]
    height2 = img2.shape[0]
    width2 = img2.shape[1]
    emptyImage = np.concatenate((img1,img2),axis=1) 
    
    for i in range(len(kp1)):
        number = random.randint(0,6)
        x1 = int(kp1[i][0])
        y1 = int(kp1[i][1])
        x2 = int(kp2[i][0])
        y2 = int(kp2[i][1])
        cv2.circle(emptyImage,(x1, y1),2,(255,0,0),2)
        cv2.circle(emptyImage,(width1+x2, y2),2,(255,0,0),2)
        cv2.line(emptyImage, (x1,y1), (width1+x2, y2), color[number], thickness=1, lineType=1)
    
    return emptyImage
    
#display1 img1 img2,height need not be same
def display1(img1, img2, kp1, kp2):
    height1 = img1.shape[0]
    width1 = img1.shape[1]
    height2 = img2.shape[0]
    width2 = img2.shape[1]
    emptyImage = np.zeros((max(height1,height2),width1+width2,3),dtype=np.uint8)
    emptyImage[0:height1,0:width1,:]=img1[0:height1,0:width1,:]
    emptyImage[0:height2,width1:width1+width2,:]=img2[0:height2,0:width2,:]
    
    for i in range(len(kp1)):
        x1 = int(kp1[i][0])
        y1 = int(kp1[i][1])
        x2 = int(kp2[i][0])
        y2 = int(kp2[i][1])
        cv2.circle(emptyImage,(x1, y1),2,(0,0,255),2)
        cv2.circle(emptyImage,(width1+x2, y2),2,(0,0,255),2)
        cv2.line(emptyImage, (x1,y1), (width1+x2, y2), (0,255,0), thickness=1, lineType=0)
    return emptyImage

#show image with key point        
def show_image(img, keypoit):
    image = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    image[:,:,:] = img[:,:,:]
    for i in range(len(keypoit)):
        x1 = int(keypoit[i][0])
        y1 = int(keypoit[i][1])
        cv2.circle(image,(x1, y1),1,(255,0,0),1)
    return image
    