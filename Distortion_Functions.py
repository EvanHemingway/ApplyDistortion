"""
Comments here
"""


import os

import numpy as np

import fnmatch

from PIL import Image

import matplotlib.pyplot as plt


# Load images into arrays and create labels
def load_distortion_data(dir_path, image_width, image_height):

    
    
    dir_path = dir_path # directory path

    image_width = image_width # image width
    image_height = image_height # image height

    dir_path = dir_path # directory path
    

    N_im = len(fnmatch.filter(os.listdir(dir_path), '*.*')) # number of all images

    N_pp = len(fnmatch.filter(os.listdir(dir_path), '*perfPersp.*')) # number of perfect perspective images
    print(f'Total images: {N_im} | Number of undistorted images: {N_pp} | Number of distorted images: {N_im - N_pp}') # print image counts
    
    
    dist_images = np.zeros((N_im,image_width,image_height)) # array of GS values for images [N x width x height]

    dist_labels = np.zeros(N_im,dtype=int) # array of labels for images denoting if they are distorted. 0 = not distorted, 1 = distored

    dist_values = np.zeros(N_im) # array of distortion coefficients

    dist_filenames = [] # list of filenames
    

    remove_idx = [] # list of indexes to remove


    # iterate over files in directory
    for kf, filename1 in enumerate(os.listdir(dir_path)):
        f1 = os.path.join(dir_path, filename1)
        # check if file is valid
        if os.path.isfile(f1) and ".jpg" in f1:
            # append filename to list
            dist_filenames.append(filename1) 
            # load image
            img = Image.open(f1)
            # convert to numpy array of GS values in range [0, 1)
            img_array = np.asarray(img)/255

            if img_array.ndim == 2:
                if img_array.shape[0] == dist_images.shape[1] and img_array.shape[1] == dist_images.shape[2]:
                    dist_images[kf,:,:] =  img_array.reshape((1,img_array.shape[0],img_array.shape[1])) # reshape to 1 x width x height
                else:
                    remove_idx.append(kf)
            if img_array.ndim == 3:
                if img_array.shape[0] == dist_images.shape[1] and img_array.shape[1] == dist_images.shape[2] and img_array.shape[2] == dist_images.shape[3]:
                    dist_images[kf,:,:,:] =  img_array.reshape((1,img_array.shape[0],img_array.shape[1],img_array.shape[2],1)) # reshape to 1 x width x height
                else:
                    remove_idx.append(kf)

            # if "perfPersp" not in filename1:
            if "perfPersp" not in filename1:
                dist_labels[kf] = 1 # label image as distorted
                # parse filename to obtain distortion coefficient
                distortion_value = int(filename1.split(".")[0].split("_")[1]) / 10**int(len(filename1.split(".")[0].split("_")[1]))
                dist_values[kf] = distortion_value
            # else, image is not distored
            else:
                dist_labels[kf] = 0 # label image as undistorted
                dist_values[kf] = 0 # distortion coefficient
        else:
            remove_idx.append(kf)


    dist_labels = dist_labels.reshape(-1,1) # reshape to 2D array

    dist_values = dist_values.reshape(-1,1) # reshape to 2D array

    # dist_images.shape

    # dist_labels.shape

    # dist_values.shape
    
    # if there are indices to be removed
    if len(remove_idx) >= 1:
        print(remove_idx)
        dist_images = np.delete(dist_images, remove_idx, 0)
        dist_labels = np.delete(dist_labels, remove_idx, 0)
        dist_values = np.delete(dist_values, remove_idx, 0)
        dist_filenames = np.delete(dist_filenames, remove_idx)
    
    
    # dist_images.shape

    # dist_labels.shape

    # dist_values.shape
    
    # place data into dict
    dist_data = {}
    dist_data["dist_images"] = dist_images
    dist_data["dist_labels"] = dist_labels
    dist_data["dist_values"] = dist_values
    dist_data["dist_filenames"] = dist_filenames
    
    # return dist_images, dist_labels, dist_values, dist_filenames
    return dist_data



# def dist_train_test_split(dist_images, dist_labels, dist_values, dist_filenames, split_amount=0.8, seed=42):
def dist_train_test_split(dist_data, split_amount=0.8, seed=42):
    
    # get data from dict
    dist_images = dist_data["dist_images"]
    dist_labels = dist_data["dist_labels"]
    dist_values = dist_data["dist_values"]
    dist_filenames = dist_data["dist_filenames"]

    # split data into train and test data

    np.random.seed(seed) # set random seed

    idx = np.random.permutation(dist_images.shape[0]) # randomized permutation of indices

    idx_split = int(split_amount*len(dist_images)) # index for the train/test

    idx_train = idx[0:idx_split] # training indices
    idx_test = idx[idx_split:] # test indices

    # print(idx_train)
    # print(idx_test)

    # split all data and labels into training and test sets
    train_images = dist_images[idx_train,:,:] # training images

    test_images = dist_images[idx_test,:,:] # test images

    train_labels = dist_labels[idx_train] # training labels

    test_labels = dist_labels[idx_test] # test labels

    train_values = dist_values[idx_train] # training distortion values

    test_values = dist_values[idx_test] # test distortion values

    # train_filenames = [filenames[idx_train[k1]] for k1 in range(0,len(idx_train))]
    # test_filenames = [filenames[idx_test[k1]] for k1 in range(0,len(idx_test))]

    train_filenames = [dist_filenames[idx] for idx in idx_train]
    test_filenames = [dist_filenames[idx] for idx in idx_test]
    
    if "dist_value_bins" in dist_data.keys():
        dist_value_bins = dist_data["dist_value_bins"]
        train_bins = dist_value_bins[idx_train]
        train_bins = dist_value_bins[idx_test]

    # print(train_labels[:,0].T)

    # print(test_labels[:,0].T)

    # print(train_values[:,0].T)

    # print(test_values[:,0].T)
    
    split_data = {}
    
    split_data["idx"] = idx
    split_data["idx_train"] = idx_train
    split_data["idx_test"] = idx_test
    
    split_data["train_images"] = train_images
    split_data["train_labels"] = train_labels
    split_data["train_values"] = train_values
    split_data["train_filenames"] = train_filenames
    
    split_data["test_images"] = test_images
    split_data["test_labels"] = test_labels
    split_data["test_values"] = test_values
    split_data["test_filenames"] = test_filenames
    
    if "dist_value_bins" in dist_data.keys():
        split_data["train_bins"] = train_bins
        split_data["test_bins"] = test_bins
    
    return split_data


def bin_dist_values(dist_data, num_bins=10, min_value=None, max_value=None):
    
    dist_values = dist_data["dist_values"]
        
    if min_value == None:
        min_value = np.min(dist_values)
    
    if max_value == None:
        max_value = np.max(dist_values)

    # calculate bin width
    bin_width = (max_value - min_value)/num_bins
    # print(bin_width)

    dist_value_bins = np.zeros((len(dist_values),1), dtype=int)

    for k1 in range(0,num_bins):
        # print(k1*bin_width)
        # print((k1+1)*bin_width)
        idx = np.where((k1*bin_width <= dist_values[:,0]) & (dist_values[:,0] <= (k1 + 1)*bin_width))
        # print(idx)
        # print(all_distortion_values[idx,0])
        dist_value_bins[idx,0] = k1

    print(dist_values.T)
    print(dist_value_bins.T)

    fig1 = plt.figure(figsize=(12,12))
    plt.plot(dist_values,dist_value_bins,"b.")
    plt.show()
    