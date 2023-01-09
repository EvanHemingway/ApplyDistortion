"""
Comments here
"""


import os

import numpy as np

import fnmatch

from PIL import Image

import matplotlib.pyplot as plt


#########################
#########################
# Load images into arrays and create labels
def load_distortion_data(dir_path, image_width, image_height, num_undist=0):
    
    """
    This function loads data from a local directory and places the data into arrays, which are then placed into a dict
    
    Inputs:
    dir_path : str
        he directory of the images to be loaded
    image_width : int
        number of pixels of image width - images with widths not matching this value will be rejected
    image_height : int
        number of pixels of image height - images with heights not matching this value will be rejected
    num_undist : int, optional
        number of undistorted images to include (the default valur of 0 is to indlude all undistorted images)
    
    Outputs:
    dist_data : dict
        dict containing several arrays, as outlined below
    dist_images : numpy float array of dimension N_im x image_width x image_height x 1
        array with image GS data
    dist_labels : numpy integer array of dimension N_im x 1
        array with image labels, with 0 for undistorted and 1 for distored
    dist_values : numpy float array of dimension N_im x 1
        array with image distortion values
    dist_filenames : list of dimension N_im
        list of filenames
        
    """

    dir_path = dir_path # directory path

    image_width = image_width # image width
    image_height = image_height # image height
    

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
    
    remove_idx = np.asarray(remove_idx, dtype=int)
    
    # if there are indices to be removed
    if len(remove_idx) >= 1:
        print(remove_idx)
        dist_images = np.delete(dist_images, remove_idx, axis=0)
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


#########################
#########################
# Remove specified amount of images matching specified label
def remove_data_by_label(dist_data, rem_label, num_remove=0, seed=42):
    
    """
    This function removes a specified number of data points that have the specified label
    
    Inputs:
    dist_data : dict
        dict containing image data, labels, distortion values, and filenames
    rem_label : int
        data point label to remove
    num_remove : int, optional
        number of data points to remove (default is 0)
    seed : int, optional
        random seed (default is 42)
    
    Outputs:
    dist_data : dict
        dict containing several arrays, as outlined below
    dist_images : numpy float array of dimension N_im x image_width x image_height x 1
        array with image GS data
    dist_labels : numpy integer array of dimension N_im x 1
        array with image labels, with 0 for undistorted and 1 for distored
    dist_values : numpy float array of dimension N_im x 1
        array with image distortion values
    dist_filenames : list of dimension N_im
        list of filenames
    
    """
    
    # dist_images = dist_data["dist_images"]
    # dist_labels = dist_data["dist_labels"]
    # dist_values = dist_data["dist_values"]
    # dist_filenames = dist_data["dist_filenames"]
    
    np.random.seed(seed) # set random seed
    
    idx = np.where(dist_data["dist_labels"][:,0] == rem_label)[0]
    # print(f"Indices matching label {rem_label}")
    # print(idx)

    idx = np.random.permutation(idx) # randomized permutation of indices
    # print(f"Permuted indices matching label {rem_label}")
    # print(idx)
    
    if num_remove >= 1 and num_remove <= len(idx):
        rem_idx = idx[0:num_remove]
        # print("Indices to remove")
        # print(rem_idx)
        
        dist_data["dist_images"] = np.delete(dist_data["dist_images"], rem_idx, 0)
        dist_data["dist_labels"] = np.delete(dist_data["dist_labels"], rem_idx, 0)
        dist_data["dist_values"] = np.delete(dist_data["dist_values"], rem_idx, 0)
        dist_data["dist_filenames"] = np.delete(dist_data["dist_filenames"], rem_idx)
        
    else:
        print("Error: Trying to remove too many indices")

    print(dist_data["dist_images"].shape)
    print(dist_data["dist_labels"].shape)
    print(dist_data["dist_values"].shape)
    print(dist_data["dist_filenames"].shape)

    return dist_data


#########################
#########################
def bin_dist_values(dist_data, num_bins=10, min_value=None, max_value=None, min_bin=False):
    
    """
    This function sorts the distortion values in dist_data["dist_values"] into discrete bins
    
    Inputs :
    dist_data : dict
        dict containing image data, labels, distortion values, and filenames
    num_bins : int, optional
        number of bins into which to sort the distortion values (default is 10)
    min_value : float, optional
        lowest bin minimum value (default is None, translating to minimum value of dist_value array)
    min_value : float, optional
        highest maximum value (default is None, translating to maximum value of dist_value array)
    min_bin : boolean, optional
        boolean that determines if there is a bin into which all values that match min_value are placed (default is False)
    
    Outputs:
    dist_data : dict
        dict containing several arrays, as outlined below
    dist_images : numpy float array of dimension N_im x image_width x image_height x 1
        array with image GS data
    dist_labels : numpy integer array of dimension N_im x 1
        array with image labels, with 0 for undistorted and 1 for distored
    dist_values : numpy float array of dimension N_im x 1
        array with image distortion values
    dist_filenames : list of dimension N_im
        list of filenames
    dist_bins : numpy integer array of dimension N_im x 1
        array of which bin the corresponding data point falls into
    
    """
    
    dist_values = dist_data["dist_values"]
        
    if min_value == None:
        min_value = np.min(dist_values)
    
    if max_value == None:
        max_value = np.max(dist_values)

    # calculate bin width
    bin_width = (max_value - min_value)/num_bins
    # print(bin_width)

    dist_value_bins = np.zeros((len(dist_values),1), dtype=int)
    
    if min_bin == False:
        # calculate bin width
        bin_width = (max_value - min_value)/num_bins
        # print(bin_width)
        print(f"Number of bins: {num_bins} | Minimum value: {min_value} | Maximum value: {max_value} | Bin Width: {bin_width}")

        for k1 in range(0,num_bins):            
            if k1 == num_bins-1:
                idx = np.where((k1*bin_width <= dist_values[:,0]) & (dist_values[:,0] <= (k1 + 1)*bin_width))
                idx = idx[0]
                print(f"Bin: {k1} | Bin range: [ {k1*bin_width} , {(k1+1)*bin_width} ] | Number of indices: {len(idx)} | Percent of examples: {len(idx)/len(dist_values)*100}")
                dist_value_bins[idx,0] = k1
            else:
                idx = np.where((k1*bin_width <= dist_values[:,0]) & (dist_values[:,0] < (k1 + 1)*bin_width))
                idx = idx[0]
                print(f"Bin: {k1} | Bin range: [ {k1*bin_width} , {(k1+1)*bin_width} ) | Number of indices: {len(idx)} | Percent of examples: {len(idx)/len(dist_values)*100}")
                dist_value_bins[idx,0] = k1
            
    if min_bin == True:
        # calculate bin width
        bin_width = (max_value - min_value)/(num_bins-1)
        # print(bin_width)
        print(f"Number of bins including min value bin: {num_bins} | Minimum value: {min_value} | Maximum value: {max_value} | Bin Width: {bin_width}")
        
        idx = np.where((dist_values[:,0] == min_value))
        idx = idx[0]
        dist_value_bins[idx,0] = 0
        
        print(f"Bin: {0} | Bin range: [ {min_value} , {min_value} ] | Number of indices: {len(idx)} | Percent of examples: {len(idx)/len(dist_values)*100}")

        for k1 in range(0,num_bins-1):
            # print(k1*bin_width)
            # print((k1+1)*bin_width)
            idx = np.where((k1*bin_width < dist_values[:,0]) & (dist_values[:,0] <= (k1 + 1)*bin_width))
            # if k1 == num_bins-1:
            #     idx = np.where((k1*bin_width <= dist_values[:,0]) & (dist_values[:,0] <= (k1 + 1)*bin_width))
            idx = idx[0]
            # print(all_distortion_values[idx,0])
            dist_value_bins[idx,0] = k1+1

            print(f"Bin: {k1+1} | Bin range: ( {k1*bin_width} , {(k1+1)*bin_width} ] | Number of indices: {len(idx)} | Percent of examples: {len(idx)/len(dist_values)*100}")

    # print(dist_values.T)
    # print(dist_value_bins.T)

    # plot
    fig1 = plt.figure(figsize=(12,4))
    plt.plot(dist_values,dist_value_bins,"b.")
    plt.xlabel("Distortion coefficient value")
    plt.ylabel("Bin")
    plt.show()
    
    fig1 = plt.figure(figsize=(12,4))
    counts, bins = np.histogram(dist_values[:,0], num_bins, (min_value, max_value))
    # print(bins)
    # print(counts)
    plt.stairs(counts, bins)
    plt.xlabel("Distortion Coefficient Bin")
    plt.ylabel("Number in bin")
    plt.show()
    
    fig1 = plt.figure(figsize=(12,4))
    # counts, bins = np.histogram(dist_value_bins[:,0], num_bins, (0, num_bins))
    # print(bins)
    # print(counts)
    # plt.stairs(counts, bins)
    for k1 in range(0,np.max(dist_value_bins[:,0])+1):
        plt.plot(k1, np.sum(dist_value_bins[:,0] == k1), "b.")
    plt.xlabel("Bin")
    plt.ylabel("Number in bin")
    plt.show()
    
    dist_data["dist_value_bins"] = dist_value_bins
    
    return dist_data


#########################
#########################
# def dist_train_test_split(dist_images, dist_labels, dist_values, dist_filenames, split_amount=0.8, seed=42):
def dist_train_test_split(dist_data, split_amount=0.8, seed=42):
    
    """
    This function splits the data into a training set and a test set, according to the user specified train/test split amount
    
    Inputs :
    dist_data : dict
        dict containing image data, labels, distortion values, and filenames
    split_amount : float, optional
        float in (0, 1) that determines the proportion of data to place into the training set (split_amount), and the proportion to place into the test set (1 - split_amount) (Default is 0.8)
    seed: int, optional
        random seed (default is 42)
    
    Outputs :
    split_data : dict
        dict containing several arrays, as outlined below
    idx : numpy integer array of dimension N_im
        array of permuted indicies
    idx_train : numpy integer array of dimension split_amount * N_im (rounded to nearest integer)
        array of permuted indices in the training set
    idx_test : numpy integer array of dimension (1 - split_amount) * N_im (rounded to nearest integer)
        array of permuted indices in the test set
    train_images : numpy float array of dimension N_im x image_width x image_height x 1
        array with image GS data
    train_labels : numpy integer array of dimension N_im x 1
        array with image labels, with 0 for undistorted and 1 for distored
    train_values : numpy float array of dimension N_im x 1
        array with image distortion values
    train_filenames : list of dimension N_im
        list of filenames
    train_bins : numpy integer array of dimension N_im x 1
        array of which bin the corresponding data point falls into
    test_images : numpy float array of dimension N_im x image_width x image_height x 1
        array with image GS data
    test_labels : numpy integer array of dimension N_im x 1
        array with image labels, with 0 for undistorted and 1 for distored
    test_values : numpy float array of dimension N_im x 1
        array with image distortion values
    test_filenames : list of dimension N_im
        list of filenames
    test_bins : numpy integer array of dimension N_im x 1
        array of which bin the corresponding data point falls into
    
    """
    
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
    train_images = dist_images[idx_train] # training images

    test_images = dist_images[idx_test] # test images

    train_labels = dist_labels[idx_train] # training labels

    test_labels = dist_labels[idx_test] # test labels

    train_values = dist_values[idx_train] # training distortion values

    test_values = dist_values[idx_test] # test distortion values

    train_filenames = [dist_filenames[idx] for idx in idx_train]
    test_filenames = [dist_filenames[idx] for idx in idx_test]
    
    train_filenames = list(np.array(dist_filenames)[idx_train])
    test_filenames = list(np.array(dist_filenames)[idx_test])
    
    if "dist_value_bins" in dist_data.keys():
        dist_value_bins = dist_data["dist_value_bins"]
        train_bins = dist_value_bins[idx_train]
        test_bins = dist_value_bins[idx_test]

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

