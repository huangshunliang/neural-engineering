#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:24:18 2019

@author: jimmy
"""
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.initializers import he_normal
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, Reshape, Permute
from keras.layers.merge import Concatenate
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers.convolutional import Cropping3D
from keras.layers import UpSampling3D
from keras.layers import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.applications import xception
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
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



class networkanalysis: #Add comments, use #%%, also break up NN and annotation codes to two files, also make pixel difference code. Either underscore or cap letter in names.
    '''
    Class for running a neural network analysis instance. 
    
    model: Input the name of a pretrained object detection neural network available in KERAS. Currently available models: 'xception', 'vgg16'
    '''
    #%% Initializes model type specified in the input 'modelname'
    def __init__(self, modelname):        
        self.modelname = modelname
               
        if self.modelname == 'xception': 
            self.model = xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        elif self.modelname == 'vgg16':
            self.model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    #%% Obtains architecture of model specified above, all layer and filter names and shapes, along with their respective indices are printed. 
        #A .png diagram of the model architecture is also outputted to the specified 'imagePath' with 'imageName' 
        
    def getarchitecture(self, imagePath, imageName):
        '''
        imagePath: Path to directory where model architecture diagram will be outputted
        imageName: Name of .png file which will be outputted. 
        '''
        self.model.summary()
        
        #Dummy data is created to input into network in order to obtain layer information
        #This section is model specific (Input dimensions are specific to model)
        if self.modelname == 'xception': 
            numpy_image = np.zeros([299,299,3])
        elif self.modelname == 'vgg16': 
            numpy_image = np.zeros([224,224,3])
        
        image_batch = np.expand_dims(numpy_image, axis=0)
        processed_image = vgg16.preprocess_input(image_batch.copy())
        
        for i, layer in enumerate(self.model.layers):
            if i != 0:
                intermediate_layer = K.function([self.model.layers[0].input],[self.model.layers[i].output]) #There are 134 layers in Xception, 22 layers in VGG16
                layer_output = intermediate_layer([processed_image])[0]
                layer_name = layer.name
                print('{} '.format(i), np.shape(layer_output), layer_name)
                
        plot_model(self.model, to_file= imagePath + '/' +  imageName, show_shapes=True, show_layer_names= True)
    #%% Extracts the activations of the specified layers when a set of video frames from 'inputPath' is passed through the network. 
        #Cosine and Euclidean distances are also obtained between each subsequent frames passed through the model for each layer specified above. 
    def extractlayers(self, inputPath, layers):
        '''
        
        inputPath: Path to directory of .png or .jpg images or frames to be passed through the model
        layers: List of layer indices to extract during analysis.
        '''
        
        startTime = time.time()
        
        self.currentDir = os.getcwd()
        self.directory = inputPath + '/'
        self.folder = self.directory.split('/')[-2]
        self.layers = layers
        
        #Initialize dictionaries in order to separate cosine and euclidean distances, layer activations, by specific layer indices being analyzed.
        self.cosine_distance_vectors = dict()
        self.euclidean_distance_vectors = dict()
        self.layer_out1 = dict()
        self.layer_out2 = dict()
        
        #Creates folders for layer activation values (format: .mat) separated by layer indices. Initializes key of layer indices for distances.
        for idx, i in enumerate(self.layers):
            os.makedirs('{}_{}_maxpool{}_layer'.format(self.folder, self.modelname, idx+1))
            self.cosine_distance_vectors['{}'.format(idx)] = []
            self.euclidean_distance_vectors['{}'.format(idx)] = []
            
        #Iterates through directory of input images/frames, applies relevant preprocessing for specified network, 
        #passes inputs to model and extracts layer activations and distances.
        for self.count, self.file in enumerate(sorted(os.listdir(self.directory))):
            self.filename = self.directory + self.file
            if self.filename.endswith('.jpg') or self.filename.endswith('.png'):
                frameStartTime = time.time()
                self.count = self.count + 1
                print('==========================================')
                print(self.count, '/', len(os.listdir(self.directory)))
                
                #This section is model specific
                if self.modelname == 'xception':
                    self.original = load_img(self.filename, target_size=(299, 299))
                elif self.modelname == 'vgg16':
                    self.original = load_img(self.filename, target_size=(224, 224))
                    
                self.numpy_image = img_to_array(self.original)
                self.image_batch = np.expand_dims(self.numpy_image, axis=0)
               
                #This section is model specific
                if self.modelname == 'xception':
                    self.processed_image = xception.preprocess_input(self.image_batch.copy())
                elif self.modelname == 'vgg16':
                    self.processed_image = vgg16.preprocess_input(self.image_batch.copy())
                for idx, i in enumerate(layers):
                    self.processlayer(idx, i)
                
                frameEndTime = time.time()  #Get frame by frame time
                print('--Frame Execution time: {} seconds--'.format(frameEndTime - frameStartTime))
        
        #Saves cosine and euclidean distances for each layer as .csv files.          
        for idx, i in enumerate(self.layers):
            if not os.path.exists(self.currentDir + '/{}_{}_maxpool{}_cosine.csv'.format(self.folder, self.modelname, idx+1)):
                np.savetxt(self.currentDir + '/{}_{}_maxpool{}_cosine.csv'.format(self.folder, self.modelname, idx+1), self.cosine_distance_vectors['{}'.format(idx)], delimiter=',')
            if not os.path.exists(self.currentDir + '/{}_{}_maxpool{}_euclidean.csv'.format(self.folder, self.modelname, idx+1)):
                np.savetxt(self.currentDir + '/{}_{}_maxpool{}_euclidean.csv'.format(self.folder, self.modelname, idx+1), self.euclidean_distance_vectors['{}'.format(idx)], delimiter=',')
        
        endTime = time.time()
        print('--Execution time: {} seconds--'.format(endTime - startTime))
        
    #%%Passes preprocessed inputs through model, extracts activations from specified layer, performs cosine and euclidean distance 
    #calculations between current input and previous input. 
    def processlayer(self, layerIndex, layer):
        with open(self.currentDir + '/{}_{}_maxpool{}_layer/{}.mat'.format(self.folder, self.modelname, layerIndex+1, self.file.split('.')[0]), 'wb') as layer_file:
            self.intermediate_layer = K.function([self.model.layers[0].input],[self.model.layers[layer].output])
            self.layer_output = self.intermediate_layer([self.processed_image])[0]
    
            if self.count < 2:
                self.layer_out1['{}'.format(layerIndex)] = self.layer_output
            if self.count >= 2:
                self.layer_out2['{}'.format(layerIndex)] = self.layer_out1['{}'.format(layerIndex)]
                self.layer_out2['{}'.format(layerIndex)] = self.layer_out2['{}'.format(layerIndex)].reshape(-1,1)
                self.layer_out1['{}'.format(layerIndex)] = self.layer_output
                self.layer_out1['{}'.format(layerIndex)] = self.layer_out1['{}'.format(layerIndex)].reshape(-1,1)
                if not os.path.exists(self.currentDir + '/{}_{}_maxpool{}_cosine.csv'.format(self.folder, self.modelname, layerIndex+1)):
                    cosineDist = spatial.distance.cosine(self.layer_out1['{}'.format(layerIndex)],self.layer_out2['{}'.format(layerIndex)])
                    print('Layer {} out of {} cosine distance: '.format(layerIndex+1,len(self.layers)), cosineDist)
                    self.cosine_distance_vectors['{}'.format(layerIndex)] = np.append(self.cosine_distance_vectors['{}'.format(layerIndex)], cosineDist)
                if not os.path.exists(self.currentDir + '/{}_{}_maxpool{}_euclidean.csv'.format(self.folder, self.modelname, layerIndex+1)):
                    euclideanDist = spatial.distance.euclidean(self.layer_out1['{}'.format(layerIndex)],self.layer_out2['{}'.format(layerIndex)])
                    print('Layer {} out of {} euclidean distance: '.format(layerIndex+1,len(self.layers)), euclideanDist)
                    self.euclidean_distance_vectors['{}'.format(layerIndex)] = np.append(self.euclidean_distance_vectors['{}'.format(layerIndex)], euclideanDist)
                
            sio.savemat(layer_file, mdict={'layer_output':self.layer_output})
            print('saved')



