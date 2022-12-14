# import tensorflow as tf

# from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt

from PIL import Image
import os

import numpy as np

import fnmatch


# Load images into arrays and create labels
def load_distortion_data(dir_path, image_width, image_height):

    filenames = [] # list of filenames

    image_width = image_width
    image_height = image_height

    # directory
    dir_path = dir_path

    N_im = len(fnmatch.filter(os.listdir(dir_path), '*.*'))

    N_pp = len(fnmatch.filter(os.listdir(dir_path), '*perfPersp.*'))
    print(f'Total images: {N_im} | Number of undistorted images: {N_pp} | Number of distorted images: {N_im - N_pp}')


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
            dist_filenames.append(filename1)
            # load image
            img = Image.open(f1)
            # convert to numpy array
            img_array = np.asarray(img)/255
            # print(img_array.shape)

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
    
    remove_idx
    
    dist_images = np.delete(dist_images, remove_idx, 0)
    
    dist_labels = np.delete(dist_labels, remove_idx, 0)
    
    dist_values = np.delete(dist_values, remove_idx, 0)
    
    dist_filenames = np.delete(dist_filenames, remove_idx)
    
    
    # dist_images.shape

    # dist_labels.shape

    # dist_values.shape
    
    dist_data = {}
    dist_data["dist_images"] = dist_images
    dist_data["dist_labels"] = dist_labels
    dist_data["dist_values"] = dist_values
    dist_data["dist_filenames"] = dist_filenames
    
    return dist_images, dist_labels, dist_values, dist_filenames



def dist_train_test_split(dist_images, dist_labels, dist_values, dist_filenames, split_amount=0.8, seed=42):

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
    
    return split_data
