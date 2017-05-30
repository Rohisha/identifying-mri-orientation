from __future__ import division
import numpy as np

import os
#from os import listdir
#from os.path import isfile, join
import nibabel as nib

from random import sample


def partition_data(data, labels):
    """return dict of dataset and labels divided into training, validation, and test sets"""
    
    n = labels.shape[0]
    # randomize ordering of data and labels
    order = sample(range(n), n)
    data_random = data[order, :, :, :]
    labels_random = labels[order]
    
    data_dict = dict()
    labels_dict = dict()
    
    # Divide into train (4/6), test (1/6) , validation (1/6) (subsample?)
    sixth = int(n/6)
    
    data_dict['X_val'] = data_random[0:(sixth), :, :, :]
    labels_dict['y_val'] = labels_random[0:(sixth)]
    data_dict['X_test'] = data_random[(sixth):(2*sixth), :, :, :]
    labels_dict['y_test'] = labels_random[(sixth):(2*sixth)]
    data_dict['X_train'] = data_random[(2*sixth):, :, :, :]
    labels_dict['y_train'] = labels_random[(2*sixth):]
    
    return data_dict, labels_dict

def extract_np_arrays_from_ADNI():
    """extract all np arrays from HCP project nii files and return as one np array"""
    
    # change this to make generalizable
    proj_path = 'ADNI: T1/' #' HCP/' # 
    proj_dirs = [f for f in os.listdir(proj_path)]
    
    num_datasets = len(proj_dirs)
    print(num_datasets)
    
    # Limit batch size to avoid memory issues
    num_datasets = 100
    
    processed_data = np.zeros(shape = [num_datasets*2*512, 256, 256, 1], dtype = 'uint16')
    processed_labels = np.zeros(shape = [num_datasets*2*512], dtype = 'int32')
    
    for i in range(2): #(len(proj_dirs)):
        proj_dir = proj_dirs[i]
        filepath = proj_path + proj_dir + '/'
        nii_file = os.listdir(filepath)[0]
        
        # Extract file data as numpy array
        # load image object
        t1w = os.path.join(filepath + nii_file)
        # get data as np array
        raw_np_data = nib.load(t1w).get_data()
        
        # Flip, slice, order randomly, pad width
        data, labels = process_raw_np(raw_np_data)
        
        # Append to processed dataset
        processed_data[i*2*512:(i*2*512) + (2*512), :, :, :] = data
        processed_labels[i*2*512:((i*2*512) + (2*512))] = labels
    
    print(processed_labels.shape)
    print(processed_data.shape)
    
    return processed_data, processed_labels


def process_raw_np(data):
    # slice data, rotate, flip, create labels (flipped/not)

    # append all 2D ~170x256 px slices together (axis=2 is the observation number axis)
    all_slices = np.append(data[:, :, :, 0], np.swapaxes(data[:, :, :, 0], 1, 2), axis = 2)
    n_slice = all_slices.shape[2]
    
    # rotate for ease of use
    all_slices = np.rot90(all_slices, axes = (0, 1))
    
    # flip each slice over x axis
    all_slices_flipped = np.flip(all_slices, axis = 1)
    
    # combined dataset with flipped (1) and original (0) slices
    slice_data = np.append(all_slices, all_slices_flipped, axis = 2)
    slice_labels = np.repeat(np.array([0, 1]), n_slice)
    
    # reorder into N x H x W x C, where:
    #    N is the number of datapoints
    #    H is the height of each image in pixels
    #    W is the height of each image in pixels
    #    C is the number of channels (usually 3: R, G, B)
    
    all_data = np.rollaxis(slice_data, 2)
    
    # Randomize data/label pairs
    order = sample(range(2*n_slice), 2*n_slice)
    all_data_random = all_data[order, :, :]
    all_labels_random = slice_labels[order]

    # Add final axis to adhere to RGB channel convention
    all_data_random = all_data[:, :, :, np.newaxis]
    
    # Pad width with 0s to make 256 x 256 instead of 256 x 170/177/180/etc.
    #   (could change architecture)
    w_pad = int((256 - all_data_random.shape[2])/2)
    all_data_random = np.pad(all_data_random, [(0, 0), (0, 0), (w_pad, w_pad), (0, 0)], mode='constant')
        
    return all_data_random, all_labels_random




    