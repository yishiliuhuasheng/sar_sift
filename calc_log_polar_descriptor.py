# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:02:00 2017

@author: Administrator
"""
import __future__
import numpy as np
import math
def calc_log_polar_descriptor(gradient,angle,x,y,scale,main_angle,d,n):
    
    cos_t = math.cos(-main_angle/180.*math.pi)
    sin_t = math.sin(-main_angle/180.*math.pi)
    
    M,N = gradient.shape
    radius = int(round(min(12*scale,min(M/2,N/2))))
    
    radius_x_left = x-radius
    radius_x_right = x+radius+1
    radius_y_up = y-radius
    radius_y_down = y+radius+1
    
    if radius_x_left <0:
        radius_x_left = 0
    if radius_x_right>=N:
        radius_x_right = N
    if radius_y_up <0:
        radius_y_up = 0
    if radius_y_down >= M:
        radius_y_down = M
    
    center_x = x-radius_x_left 
    center_y = y-radius_y_up 
    
    sub_gradient = gradient[radius_y_up:radius_y_down,radius_x_left:radius_x_right]
    sub_angle = angle[radius_y_up:radius_y_down,radius_x_left:radius_x_right]
    sub_angle = np.round((sub_angle-main_angle)*n/360)
    sub_angle[sub_angle<0] = sub_angle[sub_angle<0] + n
    sub_angle[sub_angle==0] = n

    X = list(range(-(x-radius_x_left),radius_x_right-x,1)) 
    Y = list(range(-(y-radius_y_up),radius_y_down-y,1))
    [XX,YY] = np.meshgrid(X,Y)
    c_rot = XX*cos_t - YY*sin_t
    r_rot = XX*sin_t + YY*cos_t
    
    log_angle = np.arctan2(r_rot,c_rot)
    log_angle = log_angle/math.pi*180.
    log_angle[log_angle<0] = log_angle[log_angle<0] +360
    np.seterr(divide='ignore')
    log_amplitude = np.log2(np.sqrt(np.square(c_rot)+np.square(r_rot)))
    
    log_angle = np.round(log_angle*d/360.)
    log_angle[log_angle<=0] = log_angle[log_angle<=0] + d
    log_angle[log_angle>d] = log_angle[log_angle>d] - d

    r1 = math.log(radius*0.25,2)
    r2 = math.log(radius*0.73,2)
    log_amplitude[log_amplitude<=r1] = 1   
    log_amplitude[(log_amplitude>r1) * (log_amplitude<=r2)] =2
    log_amplitude[log_amplitude>r2] = 3
    
    temp_hist = np.zeros(((2*d+1)*n,1))
    row,col = log_angle.shape
    
    for i in range(row):
        for j in range(col):
            if (i-center_y)**2+(j-center_x)**2 <=radius**2:
                angle_bin = log_angle[i,j]
                amplitude_bin = log_amplitude[i,j]
                bin_vertical = sub_angle[i,j]
                Mag = sub_gradient[i,j]
                
                if amplitude_bin==1:
                    temp_hist[int(bin_vertical)-1] =  temp_hist[int(bin_vertical)-1] + Mag
                else:
                    temp_hist[int(((amplitude_bin-2)*d+angle_bin-1)*n+bin_vertical+n)-1] = temp_hist[int(((amplitude_bin-2)*d+angle_bin-1)*n+bin_vertical+n)-1] + Mag
                
    temp_hist = temp_hist/np.sqrt(np.dot(temp_hist.T,temp_hist))
    temp_hist[temp_hist>0.2] = 0.2
    temp_hist = temp_hist/np.sqrt(np.dot(temp_hist.T,temp_hist))
    descriptor = temp_hist.reshape(-1)
    
    return descriptor
    