# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:36:38 2022

@author: ehemingway
"""

from skimage.transform import resize
import imageio.v2 as iio
import os
import numpy as np

# Define the path to the folder
directory = 'Training Set BW 1000'
folder_out = 'Training Set BW 1000 sub'

for filename in os.listdir(directory):
    image = os.path.join(directory,filename)
    img = iio.imread(directory + '/' + filename)

    img_subsampled = 255*resize(img, (240,320))
    img_sub_uint8 = img_subsampled.astype(np.uint8)
    
    # Save the processed image
    iio.imwrite(folder_out + '/' + filename,img_sub_uint8)
    