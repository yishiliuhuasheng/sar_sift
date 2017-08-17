# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:36:04 2017

@author: Administrator
"""
import __future__ 
import random
import cv2
import math
import numpy as np

iterations = 800
n = 3 #affine transform

def ransac(kp1,kp2,error_threshold):
    most_consensus_number = 0 #correspond keypoint counter
    kp_num = len(kp1)
    #list to matrix  
    kp1_matrix = np.zeros((3,kp_num), dtype="f")
    kp2_matrix = np.zeros((3,kp_num), dtype="f")
    for m in range(kp_num):
        kp1_matrix[0][m] = kp1[m][0]
        kp1_matrix[1][m] = kp1[m][1]
        kp1_matrix[2][m] = 1
        kp2_matrix[0][m] = kp2[m][0]
        kp2_matrix[1][m] = kp2[m][1]
        kp2_matrix[2][m] = 1

    for i in range(iterations):
        kp1_rand = np.zeros((n,2), dtype="f")
        kp2_rand = np.zeros((n,2), dtype="f")
        while np.array_equal(kp1_rand[0],kp1_rand[1]) or np.array_equal(kp1_rand[0],kp1_rand[2]) or np.array_equal(kp1_rand[1],kp1_rand[2]):
            for j in range(n):
                rand = random.randint(0,kp_num-1)
                kp1_rand[j,0]=kp1[rand][0]
                kp1_rand[j,1]=kp1[rand][1]
                kp2_rand[j,0]=kp2[rand][0]
                kp2_rand[j,1]=kp2[rand][1]
            
        #use 3 keypoit to compute transform matrix
        M = cv2.getAffineTransform(kp1_rand, kp2_rand)
        M_stack = np.row_stack((M, [0,0,1])) #transform matrix[a,b,c;d,e,f;0,0,1]
        #transform all kp1 by the matrix
        kp1_transform = np.dot(M_stack,kp1_matrix)
        error = kp2_matrix - kp1_transform
        error = error[0:2].transpose()
        #cumpute mean square
        mean_square = np.sqrt(np.sum(np.square(error),axis=1)/2.)
        index = np.where(mean_square < error_threshold) #return tuple,there is an array in tuple
        consensus_num = index[0].shape[0]
        #update parameter and least_mean_square
        if consensus_num > most_consensus_number:
            better_kp1 = []
            better_kp2 = []
            most_consensus_number = consensus_num
            parameter = M #select optimal parameter
            for order in range(consensus_num):
                better_kp1.append(kp1[index[0][order]])
                better_kp2.append(kp2[index[0][order]])
    return better_kp1,better_kp2

def least_square(kp1,kp2):
    kp_num = len(kp1)
    #list to matrix  
    kp1_matrix = np.zeros((3,kp_num), dtype="f")
    kp2_matrix = np.zeros((3,kp_num), dtype="f")
    for m in range(kp_num):
        kp1_matrix[0][m] = kp1[m][0]
        kp1_matrix[1][m] = kp1[m][1]
        kp1_matrix[2][m] = 1
        kp2_matrix[0][m] = kp2[m][0]
        kp2_matrix[1][m] = kp2[m][1]
        kp2_matrix[2][m] = 1

    X = kp1_matrix.T
    Y = kp2_matrix.T 
    M = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), Y) 
    S = M.T
    
    kp1_transform = np.dot(S,kp1_matrix)
    error = kp2_matrix - kp1_transform
    error = error[0:2].transpose()
        #cumpute mean square
    mean_square = np.sum(np.square(error),axis=1)
    rmse = np.sqrt(np.sum(mean_square)/float(kp_num))
    solution = S[0:2] 

    return solution,rmse