# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:36:50 2017

@author: Administrator
"""
import __future__
import numpy as np
import cv2

def image_fusion(image1, image2, solution):
    #transform image
    M = image1.shape[0]
    N = image2.shape[1]
    
    #image fusion
    pts1 = np.float32([[0,0],[0,M],[N,0],[N,M]])
    pts2 = np.float32([[0.5*N,0.5*M],[0.5*N,1.5*M],[1.5*N,0.5*M],[1.5*N,1.5*M]])
    solution_Perspective = cv2.getPerspectiveTransform(pts1,pts2) 
    image1_perspective = cv2.warpPerspective(image2, solution_Perspective, (2*N,2*M))
    #cv2.imshow("image1-perspective",image1_perspective) 
    
    solution_stack = np.row_stack((solution, [0,0,1])) #transform matrix[a,b,c;d,e,f;0,0,1]
    solution_Perspective = np.dot(solution_Perspective,solution_stack)
    #image1_transform = cv2.warpAffine(image1,solution,(2*N,2*M))
    image1_transform = cv2.warpPerspective(image1, solution_Perspective, (2*N,2*M))
    #cv2.imshow("image-transform",image1_transform) #show transform
    
    image_fusion = image1_transform + image1_perspective  
    index_same = np.where(image1_transform & image1_perspective)
    row = index_same[0]
    colm = index_same[1]   
    image_fusion[row,colm] = image1_transform[row,colm]/2. + image1_perspective[row,colm]/2.     
     
    return image_fusion
    
def common_region(image1,image2,solution):
     #transform image
    M = image1.shape[0]
    N = image2.shape[1]
    
    solution_stack = np.row_stack((solution, [0,0,1])) #transform matrix[a,b,c;d,e,f;0,0,1]
    common1 = cv2.warpPerspective(image1, solution_stack, (N,M))
    common2 = image2-(image2-common1)
    return common1,common2