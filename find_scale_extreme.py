# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:43:56 2017

@author: Administrator
"""
import __future__
import numpy as np
import math

import calculate_oritation_hist

def find_scale_extreme(sar_harris_function,threshold,sigma,ratio,gradient,angle):
    M,N,num = sar_harris_function.shape
    BORDER_WIDTH = 1
    HIST_BIN = 36
    SIFT_ORI__PEAK_RATIO = 0.8
    key_number = 0
    key_point_array = np.zeros((M*N,num-2))
    for i in range(num):
        temp_current = sar_harris_function[:,:,i]
        gradient_current = gradient[:,:,i]
        angle_current = angle[:,:,i]
        for j in range(BORDER_WIDTH,M-BORDER_WIDTH,1):
            for k in range(BORDER_WIDTH,N-BORDER_WIDTH,1):
                temp = temp_current[j,k]
                if temp>threshold \
                   and temp>temp_current[j-1,k-1] and temp>temp_current[j-1,k] and temp>temp_current[j-1,k+1] \
                   and temp>temp_current[j,k-1] and temp>temp_current[j,k+1] \
                   and temp>temp_current[j+1,k-1] and temp>temp_current[j+1,k] and temp>temp_current[j+1,k+1] :
                       scale = sigma*ratio**(i+1)
                       hist,max_value = calculate_oritation_hist.calculate_oritation_hist(k,j,scale,gradient_current,angle_current,HIST_BIN)
                       
                       mag_thr = max_value*SIFT_ORI__PEAK_RATIO
                       for kk in range(HIST_BIN):
                           if kk ==0:
                               k1=HIST_BIN-1
                           else:
                               k1 = kk-1
                           if kk==HIST_BIN-1:
                               k2 = 0
                           else:
                               k2 = kk+1
                           if hist[kk]>hist[k1] and hist[kk]>hist[k2] and hist[kk]>mag_thr:
                               bins = kk + 0.5*(hist[k1]-hist[k2])/float((hist[k1]+ hist[k2] - 2*hist[kk]))
                               if bins <0:
                                   bins = HIST_BIN+bins
                               elif bins>=HIST_BIN:
                                   bins = bins-HIST_BIN
                               key_number = key_number + 1    
                               key_point_array[key_number,0] = k
                               key_point_array[key_number,1] = j
                               key_point_array[key_number,2] = sigma*ratio**(i)
                               key_point_array[key_number,3] = i
                               key_point_array[key_number,4] = (360/float(HIST_BIN))*bins
                               key_point_array[key_number,5] = hist[kk]
    key_point_array = key_point_array[1:key_number,0:6]
    return key_point_array