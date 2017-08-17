# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:34:36 2017

@author: Administrator
"""
import __future__
import cv2
import time
import numpy as np

import build_scale
import find_scale_extreme
import calc_descriptors
import match
import display
import ransac
import image_fusion
import MI

#read image
img1 = cv2.imread("testdata/h1.jpg") # image to be registered
img2 = cv2.imread("testdata/h2.jpg") # reference image

#rgb2gray
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gray1 = gray1/255.
gray2 = gray2/255.

#initial parameter
time1 = time.time()
sigma = 2                      #initial layer scale
ratio = 2**(1/3.)               #scale ratio
Mmax = 8                       #layer number
d = 0.04
d_SH_1 = 0.8                   #Harros function threshold
d_SH_2 = 0.8                   #Harros function threshold
distRatio = 0.9
error_threshold = 1

#Creat sar-harris function
sar_harris_function_1,gradient_1,angle_1 = build_scale.build_scale(gray1,sigma,Mmax,ratio,d)
sar_harris_function_2,gradient_2,angle_2 = build_scale.build_scale(gray2,sigma,Mmax,ratio,d)
time_harris_function = time.time()
print("Create SAR HARRIS function Spend time:",time_harris_function-time1)

#Feaarure point detection
GR_key_array_1 = find_scale_extreme.find_scale_extreme(sar_harris_function_1,d_SH_1,sigma,ratio,gradient_1,angle_1)
GR_key_array_2 = find_scale_extreme.find_scale_extreme(sar_harris_function_2,d_SH_2,sigma,ratio,gradient_2,angle_2)
time_point = time.time()
print("Feature point detection:", time_point-time_harris_function)

#calculating descriptors
descriptors_1, locs_1 = calc_descriptors.calc_descriptors(gradient_1,angle_1,GR_key_array_1)
descriptors_2, locs_2 = calc_descriptors.calc_descriptors(gradient_2,angle_2,GR_key_array_2)
time_descriptor = time.time()
print("calculating descriptor:", time_descriptor-time_point)

#match
kp1,kp2,des1,des2 = match.delete_duplications(GR_key_array_1[:,0:2],GR_key_array_2[:,0:2],descriptors_1,descriptors_2)
good_kp1,good_kp2 = match.deep_match(kp1,kp2,des1,des2,distRatio)
better_kp1,better_kp2 = ransac.ransac(good_kp1,good_kp2,error_threshold)
solution1,rmse = ransac.least_square(better_kp1,better_kp2)

#compute MI
common1,common2 = image_fusion.common_region(gray1,gray2,solution1)
mi = MI.MI(common1,common1)

#display
image_match = display.display(img1,img2,better_kp1,better_kp2)
image_fusion = image_fusion.image_fusion(img1,img2,solution1)
time_registration = time.time()
print("image registration:", time_registration-time_descriptor)
time2 = time.time()
print("total time:",time2 - time1)

cv2.imshow("image_match",image_match)
cv2.imshow("image_fusion",image_fusion)
cv2.waitKey(0)