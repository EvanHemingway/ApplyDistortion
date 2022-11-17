# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:06:29 2022

@author: ehemingway
"""

Create folder "Training set"
Get 10 random k's from -0.33 to 0'

For each k 
go into Raw Images and iterate through each image. Also, get inverse params.
Apply inverse mapping to 640x480 grid to get float locations, where 640x480 grid CS is normalized by r=1 on diagonal.
Bilinearly interpolate to sample RGB values from 1280x960 raw image.
Render (save) the 640 x 480 image as .jpg with k value in name in Training set folder.
