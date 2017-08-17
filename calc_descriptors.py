# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:50:28 2017

@author: Administrator
"""
import __future__
import numpy as np
import calc_log_polar_descriptor

def calc_descriptors(gradient,angle,key_point_array):
    circle_bin = 8
    LOG_DESC_HIST_BINS = 8
    
    M = key_point_array.shape[0]
    d = circle_bin
    n = LOG_DESC_HIST_BINS
    descriptors = np.zeros((M,(2*d+1)*n))
    locs = key_point_array
    
    for i in range(M):
        x = int(key_point_array[i,0])
        y = int(key_point_array[i,1])
        scale = key_point_array[i,2]
        layer = int(key_point_array[i,3])
        main_angle = key_point_array[i,4]
        current_gradient = gradient[:,:,layer]
        current_angle = angle[:,:,layer]
        descriptors[i,:] = calc_log_polar_descriptor.calc_log_polar_descriptor(current_gradient,current_angle,x,y,scale,main_angle,d,n)
            
    return descriptors,locs