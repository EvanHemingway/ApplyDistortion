# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:06:29 2022

@author: ehemingway
"""

import random
from findInverseModule import findInverse as findInv
import imageio.v2 as iio
from math import floor as floor
from math import ceil as ceil
import numpy as np

def persp2barrel(k,img):
    undist = findInv(k)
    c = 0
    img_dist = np.array([0]*640*480*3).reshape(480,640,3).astype(np.uint8)
    for x_orig in range(-320,320):
        x_orig = x_orig + 0.5
        r = 0
        for y_orig in range(-240,240):
            y_orig = y_orig + 0.5
            r_orig = (x_orig**2+y_orig**2)**0.5/(319**2+239**2)**0.5
            scale = (1 + undist[0] * r_orig
                               + undist[1] * r_orig**2
                               + undist[2] * r_orig**3
                               + undist[3] * r_orig**4
                               + undist[4] * r_orig**5)
            x_new = x_orig*scale
            y_new = y_orig*scale
            
            interp_up = img[ceil(y_new)+480,floor(x_new)+640,:]*(ceil(x_new)-x_new)+img[ceil(y_new)+480,ceil(x_new)+640,:]*(x_new-floor(x_new))
            interp_low = img[floor(y_new)+480,floor(x_new)+640,:]*(ceil(x_new)-x_new)+img[floor(y_new)+480,ceil(x_new)+640,:]*(x_new-floor(x_new))
      
            interp = (ceil(y_new)-y_new)*interp_low + (y_new-floor(y_new))*interp_up
            interp_int = [int(interp[0]), int(interp[1]), int(interp[2])]
            img_dist[r,c] = interp_int 
            r = r + 1
        c = c + 1
    
    return img_dist

for i in range(1,26):
    k = random.randint(1,100)/100*-0.2
    k = np.float32(k)
    img = iio.imread('Raw Images BW/' + str(i) + '.jpg')
    img_dist = persp2barrel(k,img)
    iio.imwrite('Training Set BW/'+str(i)+'_'+'k-0p'+str(k).split('.')[1]+'.jpg',img_dist)
    
for i in range(26,51):
    k = 0
    img = iio.imread('Raw Images BW/' + str(i) + '.jpg')
    img_dist = persp2barrel(k,img)
    iio.imwrite('Training Set BW/'+str(i)+'_perfPersp.jpg',img_dist)