# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:25:56 2017

@author: Administrator
"""
import __future__
import numpy as np
import math
def calculate_oritation_hist(x,y,scale,gradient,angle,n):
    M,N = gradient.shape
    radius = int(round(min(6*scale,min(M/2,N/2))))
    sigma = 2*scale
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
    
    #X = list(range(-(x-radius_x_left),radius_x_right-x,1)) 
    #Y = list(range(-(y-radius_y_up),radius_y_down-y,1))
    #[XX,YY] = np.meshgrid(X,Y)
    W = sub_gradient
    bins = np.round(sub_angle*n/360.)
    
    bins[bins>=n] = bins[bins>=n]-n
    bins[bins<0] = bins[bins<0]+n

    tem_hist = np.zeros((n,1))
    row,col = bins.shape

    for i in range(row):
        for j in range(col):
            if (i-center_y)**2+(j-center_x)**2 <=radius**2:
                tem_hist[int(bins[i,j])] = tem_hist[int(bins[i,j])] + W[i,j]

    

#smooth histogram
    hist = np.zeros((n,1))
    hist[0] = (tem_hist[34] + tem_hist[2])/16. + 4*(tem_hist[35]+tem_hist[1])/16. + tem_hist[0]*6/16. 
    hist[1] = (tem_hist[35] + tem_hist[3])/16. + 4*(tem_hist[0]+tem_hist[2])/16. + tem_hist[1]*6/16.
    hist[2:n-2] = (tem_hist[0:n-4] + tem_hist[4:n])/16. + 4*(tem_hist[1:n-3]+tem_hist[3:n-1])/16. + tem_hist[2:n-2]*6/16.
    hist[n-2] = (tem_hist[n-4] + tem_hist[0])/16. + 4*(tem_hist[n-3]+tem_hist[n-1])/16. + tem_hist[1]*6/16.
    hist[n-1] = (tem_hist[n-3] + tem_hist[1])/16. + 4*(tem_hist[n-1]+tem_hist[0])/16. + tem_hist[n-1]*6/16.
    
    max_value = np.amax(hist)
    
    return hist,max_value

if __name__ == '__main__':
    a = np.ones((20,20))
    b = np.ones((20,20))
    m,n = calculate_oritation_hist(18,4,2,a,b,36)



    