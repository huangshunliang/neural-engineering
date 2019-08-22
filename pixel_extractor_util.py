#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:23:12 2019

@author: jimmy
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from keras import backend as K
from sklearn.metrics.pairwise import *
from scipy import spatial
import scipy.io as sio
import os
import csv
import cv2
from matplotlib.animation import FuncAnimation
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
import time

#%%Extract pixel values from folder of frames, find cosine distance and euclidean distance
            
def pixelAnalysis(in_path, outputPixels = False):
    '''
    
    in_path: path to folder of images or frames to be analyzed
    outputPixels: False (Default), if True, the actual pixel values will also be outputted
    '''

    euclidean_vector = np.array([])    
    cosine_vector = np.array([])   
    directory = in_path + '/'
    
    main_out_path = '{}_pixel_analysis'.format(in_path.split('/')[-1])
    pixel_out = main_out_path + '/pixel_values'
    os.mkdir(main_out_path)
    os.mkdir(pixel_out)
    
    
    for count, file in enumerate(sorted(os.listdir(directory))):
        filename = directory + file
        count = count + 1
        print('==============================================')
        print(count, '/',len(os.listdir(directory)))
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        numpy_image =  gray
        resolution = np.shape(numpy_image)
        print(resolution)
        pixel_vector = numpy_image
        numpy_image = numpy_image.reshape(-1,1)
        
        if count < 2:
            frame_out1 = numpy_image
        if count >= 2:
            frame_out2 = frame_out1
            frame_out1 = numpy_image
    
            euclideanDist = distance.euclidean(frame_out1, frame_out2)
            cosineDist = distance.cosine(frame_out1, frame_out2)
            
            euclidean_vector = np.append(euclidean_vector,  euclideanDist)
            cosine_vector = np.append(cosine_vector, cosineDist)
        
        if outputPixels == True:    
            with open(pixel_out + '/{}_pixels.mat'.format(file.split('.')[0]), 'wb') as outfile:    
                sio.savemat(outfile, mdict={'pixel_values':pixel_vector})
            
    np.savetxt(pixel_out+"{}_pixel_euclidian_vector.csv".format(in_path.split('/')[-1]), euclidean_vector, delimiter=",")
    np.savetxt(pixel_out+"{}_pixel_cosine_vector.csv".format(in_path.split('/')[-1]), cosine_vector, delimiter=",")
                    
            
            
