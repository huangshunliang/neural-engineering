#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:20:30 2019

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
import pandas as pd
from matplotlib.animation import FuncAnimation
import time


#%%Copies jpg from one folder into another. Useful if there are other types of files in the input folder (CHECKED)
def jpg_Folder2Folder(dir_in,dir_out): 
    import os
    from shutil import copyfile
    directory = os.fsencode(dir_in)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"): 
            copyfile(dir_in+'/{}'.format(filename), 
                     dir_out + '/{}'.format(filename))
    print('DONE')

#%%Copies csv from one folder into another. Useful if there are other types of files in the input folder (CHECKED)            
def csv_Folder2Folder(dir_in,dir_out): 
    import os
    from shutil import copyfile
    directory = os.fsencode(dir_in)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            copyfile(dir_in+'/{}'.format(filename), 
                     dir_out + '/{}'.format(filename))
            
#%%Copies and separates jpg and txt from a folder into two respective output folders (CHECKED)
def jpg_txt_Folder2Folder(dir_in, dir_out_jpg, dir_out_txt): 
    import os
    from shutil import copyfile
    directory = os.fsencode(dir_in)

    for file in os.listdir(directory):
         filename = os.fsdecode(file)
         if filename.endswith(".jpg"): 
             copyfile(dir_in + '/{}'.format(filename), 
                      dir_out_jpg + '/{}'.format(filename)) #seg_ for non-labeled images  
         elif filename.endswith(".txt"): 
             copyfile(dir_in + '/{}'.format(filename), 
                      dir_out_txt + '/{}'.format(filename))
             
#%%Input directory of pixel accurate CSV outputs with headers, and outputs textfiles readable by annotation GUI
def convertPIXELcsv2GUItxt(dir_in, dir_out, class_in): 
    import pandas as pd
    import os
    from shutil import copyfile

    in_direct = dir_in + '/'
    directory = os.fsencode(in_direct)
    out_direct = dir_out + '/'
    #Reference class label file and create a dictionary of labels and label values 
    class_file = open(class_in)
    class_dict = dict()
    for i, line in enumerate(class_file):
        class_dict.update({i : line.strip()})
    
    class_file.close()    
    #Create initial dataframe on Pandas for YOLOV3 output CSVs
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            data = pd.read_csv(in_direct+filename)
            df = pd.DataFrame(data, columns = ['Class', 'x start', 'y start', 'x end', 'y end'])
            imageName = os.path.split(filename)[-1].split('.')[0]
    #Create .txt inputs for labeling
            
            with open(out_direct+'{}.txt'.format(imageName),'w', encoding="utf-8") as txtfile:
                print("##################################################### \n")
                print(len(df))
                idx = 0
                while(idx<len(df)):
                    xStart = df['x start'][idx]
                    yStart = df['y start'][idx]
                    xEnd = df['x end'][idx]
                    yEnd = df['y end'][idx]
                    print(xStart, " ", yStart, " ", xEnd, " ", yEnd, "\n")
                    txtfile.write('%s.jpg ' % imageName)
                    txtfile.write((class_dict[df['Class'][idx]]))
                    txtfile.write(' ' + (str(xStart)))
                    txtfile.write(' ' + (str(yStart)))
                    txtfile.write(' ' + (str(xEnd)))
                    txtfile.write(' ' + (str(yEnd))+'\n')
                    
#%%Converts label output of the annotation GUI into csv format for analysis (CHECKED)
def GUItxt2ANALYSIScsv(dir_in, dir_out, dir_classes): 
    import os
    
    class_file = open(dir_classes)
    class_dict = dict()
    for i, line in enumerate(class_file):
        class_dict.update({line.strip() : i})
    class_file.close()
    
    in_directory = dir_in + '/'
    in_dir = os.fsencode(in_directory)
    out_directory =  dir_out + '/'
    out_dir = os.fsencode(out_directory)

    for num, file in enumerate(sorted(os.listdir(in_dir))):
        print(num+1,"/",len(os.listdir(in_dir)))
        print(file,"\n")
        filename = os.fsdecode(file)
        if filename.endswith('.txt'):
            imageName = os.path.split(filename)[-1].split('.')[0]
            with open(out_directory+imageName+'.csv','a') as csv_out:
                header = "Class" + "," + "x start" + "," + "y start" + "," + "x end" + "," + "y end" + "\n"
                csv_out.write(header)
            with open(in_directory + filename, encoding="utf-8") as f:
                for (i, line) in enumerate(f):
                    tmp2 = [t.strip() for t in line.split()]
                    tmp = [float(t) if idx > 0 else t for idx, t in enumerate(tmp2[-5:])]
                
                    with open(out_directory + imageName+'.csv','a') as csv_out:
                        csv_out.write("{}".format(class_dict[tmp[0]]))
                        csv_out.write(",{}".format(tmp[1]))
                        csv_out.write(",{}".format(tmp[2]))
                        csv_out.write(",{}".format(tmp[3]))
                        csv_out.write(",{}".format(tmp[4]))
                        csv_out.write("\n")

#%%
def vectorAnalysis(dir_in, dir_out, dir_classes, thresh = 50):
    '''
    Input label outputs from annotation GUI and output semantics euclidean/cosine distances 
    and position euclidean/cosine distances
    
    dir_in: directory of GUI output labels
    dir_out: directory where you would like to deposit files
    dir_classes: path to classes file you would like to reference
    thresh: Specify an integer threshold for filtering out position eucilidean and cosine distance spikes 
    '''
    
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import pandas as pd
    import scipy.io as sio
    from scipy.spatial import distance
    
    class_file = open(dir_classes) 
    class_dict = dict()
    for i, line in enumerate(class_file): #Creates label classes dictionary for reference
        class_dict.update({line.strip() : i})
    class_file.close()
    
    in_directory = dir_in + '/'
    in_dir = os.fsencode(in_directory)
    out_directory =  dir_out + '/'
    
    label_out = out_directory + 'label_CSVs' #Directory for label CSVs
    vector_out = out_directory + 'vector_files' #Directory for vector files
    
    os.mkdir(label_out) #Creates folders for CSVs, vectors, and euclidean/cosine distance vectors 
    os.mkdir(vector_out)
    
    label_out = label_out + '/'
    vector_out = vector_out + '/'
    
    #Iterates through files in annotation labels folder and extracts label text files, and converts to CSV
    for num, file in enumerate(sorted(os.listdir(in_dir))): 
        print(num+1,"/",len(os.listdir(in_dir)))
        print(file,"\n")
        filename = os.fsdecode(file)
        if filename.endswith('.txt'):
            imageName = os.path.split(filename)[-1].split('.')[0]
            with open(label_out+imageName+'.csv','a') as csv_out:
                header = "Class" + "," + "x start" + "," + "y start" + "," + "x end" + "," + "y end" + "\n"
                csv_out.write(header)
            with open(in_directory + filename, encoding="utf-8") as f:
                for (i, line) in enumerate(f):
                    tmp2 = [t.strip() for t in line.split()]
                    tmp = [float(t) if idx > 0 else t for idx, t in enumerate(tmp2[-5:])]
                
                    with open(label_out + imageName+'.csv','a') as csv_out:
                        csv_out.write("{}".format(class_dict[tmp[0]]))
                        csv_out.write(",{}".format(tmp[1]))
                        csv_out.write(",{}".format(tmp[2]))
                        csv_out.write(",{}".format(tmp[3]))
                        csv_out.write(",{}".format(tmp[4]))
                        csv_out.write("\n")

    
    vect = np.zeros((len(class_dict), len(os.listdir(label_out)))) #Initialize semantics only vector
    
    #Iterates through all converted CSV files and extracts semantic information
    for num, file in enumerate(sorted(os.listdir(label_out))): 
        print("##################################################### \n")
        print(num+1,"/",len(os.listdir(label_out)))
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            df = pd.read_csv(label_out+filename)
            for row in range(0,len(df)):
                vect[int(df['Class'][row])][num]+=1 #Generates sparse matrix of class label instances where row index represents respective label classes.
    
    sio.savemat(vector_out+"{}_semantics_vect.mat".format(in_directory.split('/')[-2]), mdict={'semantics_vector':vect})
    
    pos_vect = np.zeros((len(class_dict), len(os.listdir(label_out)),int((vect.max()*2)+1))) #Initialize position matrix
    
    for num, file in enumerate(sorted(os.listdir(label_out))): #Iterates through all converted CSV files and extracts positional information
        print("##################################################### \n")
        print(num+1,"/",len(os.listdir(label_out)))
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            df = pd.read_csv(label_out+filename)
            print(df)
            
            #Iterates within each label CSV to extract position information, and calculate centroid of bounding boxes
            for idx, row in df.iterrows(): 
                pos_vect[int(row['Class'])][num][0]+=1
                centroid_x = (row['x start'] + row['x end'])/2
                centroid_y = (row['y start'] + row['y end'])/2
                print(type(((pos_vect[int(row['Class'])][num][0])*2)-1)) ###For debugging purposes###
                print(((pos_vect[int(row['Class'])][num][0])*2)) ###For debugging purposes###
                pos_vect[int(row['Class'])][num][int(((pos_vect[int(row['Class'])][num][0])*2)-1)] = centroid_x
                pos_vect[int(row['Class'])][num][int(((pos_vect[int(row['Class'])][num][0])*2))] = centroid_y
                
    
    sio.savemat(vector_out +"{}_positions_vect.mat".format(in_directory.split('/')[-2]), mdict={'positions_vector':pos_vect})

    semantic_euc = np.array([]) #Initialize vector for semantic euclidean distance
    semantic_cos = np.array([]) #Initialize vector for semantic cosine distance
    
    for i in range(0,len(os.listdir(label_out))-1): #Iterate through semantic vector, and perform euclidean and cosine distance calculations
        a = vect[:,i]
        b = vect[:,i+1]
        semDst = distance.euclidean(a,b)
        cosDst = distance.cosine(a,b)
        semantic_euc = np.append(semantic_euc, semDst)
        semantic_cos = np.append(semantic_cos, cosDst)
    
    sio.savemat(vector_out+"{}_euclidean_semantics.mat".format(in_directory.split('/')[-2]), mdict={'euclidean_semantics':semantic_euc})
    sio.savemat(vector_out+"{}_cosine_semantics.mat".format(in_directory.split('/')[-2]), mdict={'cosine_semantics':semantic_cos})


    position_euc = np.zeros((len(class_dict), len(os.listdir(label_out))-1,int(vect.max())+1)) #Initialize positional euclidean distance matrix
    position_cos = np.zeros((len(class_dict), len(os.listdir(label_out))-1,int(vect.max())+1)) #Initialize positional cosine distance matrix

    for i in range(0,len(os.listdir(label_out))-1): #Iterate through frames axis of position vector
        print('##################################################### \n')
        print('Frame ',i+1,' / ',len(os.listdir(label_out))-1)
        for j in range(0, len(class_dict)): #Iterate through label classes axis of position vector
            euctemppositions = []
            costemppositions = []
            if pos_vect[j][i][0] == 0:
                continue
            
            for idx, k in enumerate(pos_vect[j][i]): #Iterate through centroids axis of position vector 
                if idx == 0:
                    position_euc[j][i][0] = k
                    position_cos[j][i][0] = k
                elif idx > 0: 
                    if (idx % 2 != 0):
                        next_pos_num = 1 #Initialized at 1, to be incremented in upcoming while-loop
                        euctemp = []
                        costemp = []
                        if (pos_vect[j][i][idx] == 0) and (pos_vect[j][i][idx+1] == 0): #Assumption made here is that centroids (if exists) will never be at 0,0.
                            break
                            
                        while (pos_vect[j][i][idx] != 0) and (pos_vect[j][i][idx+1] != 0) and (pos_vect[j][i+1][(next_pos_num*2)-1] != 0) and (pos_vect[j][i+1][(next_pos_num*2)] != 0):
                            a = pos_vect[j][i][idx:idx+2]
                            b = pos_vect[j][i+1][int((next_pos_num*2)-1):int((next_pos_num*2)+1)]

                            eucpos_dst = distance.euclidean(a,b)
                            cospos_dst = distance.cosine(a,b)
                            euctemp.append(eucpos_dst)
                            costemp.append(cospos_dst)
                            next_pos_num+=1
                            if (next_pos_num > vect.max()):
                                break
                        if (pos_vect[j][i][idx] != 0) and (pos_vect[j][i][idx+1] != 0) and (pos_vect[j][i+1][((next_pos_num-1)*2)-1] != 0) and (pos_vect[j][i+1][((next_pos_num-1)*2)] != 0):
                            euctemppositions = euctemppositions + [euctemp] #Constructs a tuple containing every combination of distances
                            costemppositions = costemppositions + [costemp]
                        
            while(len(euctemppositions) > 0):
                idx_min_a = euctemppositions.index(min(euctemppositions)) #index of current frame comparison (corresponds to outer index)
                idx_min_b = euctemppositions[idx_min_a].index(min(min(euctemppositions))) #index of the next frame comparison (corresponds to inner index)
                min_dst = min(min(euctemppositions))
                position_euc[j][i][idx_min_a+1] = min_dst #Assign minimum distance to corresponding positional object
            
                del euctemppositions[idx_min_a] #deletes index of current frame comparison (outer index) after computation
                    
                for pos in range(0,len(euctemppositions)): #Iterates through indices of next frame comparisons (inner index) and deletes after computation
                    del euctemppositions[pos][idx_min_b]

                euctemppositions = [position for position in euctemppositions if position]
                
            while(len(costemppositions) > 0):
                idx_min_a = costemppositions.index(min(costemppositions)) #index of current frame comparison (corresponds to outer index)
                idx_min_b = costemppositions[idx_min_a].index(min(min(costemppositions))) #index of the next frame comparison (corresponds to inner index)
                min_dst = min(min(costemppositions))
                position_cos[j][i][idx_min_a+1] = min_dst #Assign minimum distance to corresponding positional object
            
                del costemppositions[idx_min_a] #deletes index of current frame comparison (outer index) after computation
                    
                for pos in range(0,len(costemppositions)): #Iterates through indices of next frame comparisons (inner index) and deletes after computation
                    del costemppositions[pos][idx_min_b]

                costemppositions = [position for position in costemppositions if position]
    
    position_euc_percentiles = np.zeros((len(os.listdir(label_out))-1))
    position_cos_percentiles = np.zeros((len(os.listdir(label_out))-1))
    
    for i in range(0,len(os.listdir(label_out))-1):
        eucclean = []
        cosclean = []
        print('##################################################### \n')
        print('Frame ',i+1,' / ',len(os.listdir(label_out))-1)
        for j in range(0, len(class_dict)):
            for idx, k in enumerate(position_euc[j][i]):
                if idx > 0:
                    if not np.isnan(k):
                        eucclean.append(k)
            for idx, k in enumerate(position_cos[j][i]):
                if idx > 0:
                    if not np.isnan(k):
                        cosclean.append(k)
        
        eucclean = np.trim_zeros(eucclean)   
        cosclean = np.trim_zeros(cosclean)                  
        if len(eucclean) == 0:
            position_euc_percentiles[i] = 0
            print("length is zero")
        else:    
            position_euc_percentiles[i] = np.percentile(eucclean,95)
            
        if np.diff(position_euc_percentiles[i-1:i+1]) > thresh: #Specified spike filtering threshold
            position_euc_percentiles[i] = position_euc_percentiles[i-1]
        if position_euc_percentiles[i] != 0:
            print("not zero")
            
        if len(cosclean) == 0:
            position_cos_percentiles[i] = 0
            print("length is zero")
        else:    
            position_cos_percentiles[i] = np.percentile(cosclean,95)
    
    sio.savemat(vector_out+"{}_euclidean_positions.mat".format(in_directory.split('/')[-2]), mdict={'euclidean_positions':position_euc_percentiles})
    sio.savemat(vector_out+"{}_cosine_positions.mat".format(in_directory.split('/')[-2]), mdict={'cosine_positions':position_cos_percentiles})
    
    #%% Class implementation test
    
class annAnalyze():
    #Initialize vectorAnalysis class, during initialization, CSV formatted outputs from the GUI are created and dumped into a folder in "dir_out"
    def __init__(self, dir_in, dir_out, dir_classes):
        
        '''
        Input label outputs from annotation GUI and output semantics euclidean/cosine distances 
        and position euclidean/cosine distances. ***This is the object oriented implementation, 
        so you can pick and choose which exact metrics you wish you extract.***
        
        dir_in: directory of GUI output labels
        dir_out: directory where you would like to deposit files
        dir_classes: path to classes file you would like to reference
        '''
        
        import os
        import sys
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import pandas as pd
        import scipy.io as sio
        from scipy.spatial import distance
        
        self.vector_type = []
        
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.dir_classes = dir_classes
        
        self.in_directory = self.dir_in + '/'
        self.in_dir = os.fsencode(self.in_directory)
        self.out_directory =  self.dir_out + '/'
        
        self.label_out = self.out_directory + 'label_CSVs' #Directory for label CSVs
        if not os.path.exists(self.label_out):
            os.mkdir(self.label_out) #Creates folders for label CSVs
        self.label_out = self.label_out + '/'
        
        self.class_file = open(self.dir_classes) 
        self.class_dict = dict()
        for i, line in enumerate(self.class_file): #Creates label classes dictionary for reference
            self.class_dict.update({line.strip() : i})
        self.class_file.close()
        
        #Iterates through files in annotation labels folder and extracts label text files, and converts to CSV
        self.printProgressBar(0, len(os.listdir(self.in_dir)), prefix = 'Converting labels:', suffix = 'Complete', length = 50)
        for num, file in enumerate(sorted(os.listdir(self.in_dir))): 
            #print(num+1,"/",len(os.listdir(self.in_dir)))
            #print(file,"\n")
            self.printProgressBar(num + 1, len(os.listdir(self.in_dir)), prefix = 'Converting labels:', suffix = 'Complete', length = 50)
            
            
            filename = os.fsdecode(file)
            if filename.endswith('.txt'):
                imageName = os.path.split(filename)[-1].split('.')[0]
                with open(self.label_out+imageName+'.csv','a') as csv_out:
                    header = "Class" + "," + "x start" + "," + "y start" + "," + "x end" + "," + "y end" + "\n"
                    csv_out.write(header)
                with open(self.in_directory + filename, encoding="utf-8") as f:
                    for (i, line) in enumerate(f):
                        tmp2 = [t.strip() for t in line.split()]
                        tmp = [float(t) if idx > 0 else t for idx, t in enumerate(tmp2[-5:])]
                    
                        with open(self.label_out + imageName+'.csv','a') as csv_out:
                            csv_out.write("{}".format(self.class_dict[tmp[0]]))
                            csv_out.write(",{}".format(tmp[1]))
                            csv_out.write(",{}".format(tmp[2]))
                            csv_out.write(",{}".format(tmp[3]))
                            csv_out.write(",{}".format(tmp[4]))
                            csv_out.write("\n")
                            
        self.vect = np.zeros((len(self.class_dict), len(os.listdir(self.label_out)))) #Initialize semantics-only vector
        
        #Iterates through all converted CSV files and extracts semantic information
        self.printProgressBar(0, len(os.listdir(self.label_out)), prefix = 'Extracting labels:', suffix = 'Complete', length = 50)
        for num, file in enumerate(sorted(os.listdir(self.label_out))): 
            #print("##################################################### \n")
            #print(num+1,"/",len(os.listdir(self.label_out)))
            self.printProgressBar(num + 1, len(os.listdir(self.label_out)), prefix = 'Extracting labels:', suffix = 'Complete', length = 50)
            filename = os.fsdecode(file)
            if filename.endswith('.csv'):
                df = pd.read_csv(self.label_out+filename)
                for row in range(0,len(df)):
                    self.vect[int(df['Class'][row])][num]+=1 #Generates sparse matrix of class label instances where row index represents respective label classes.
                            
                        
                    
                            
    def extractSemantics(self):
        
        self.vector_type = self.vector_type + ['s']
        
        self.vector_out = self.out_directory + 'vector_files' #Directory for vector files
        if not os.path.exists(self.vector_out):
            os.mkdir(self.vector_out)
        self.vector_out = self.vector_out + '/'     
        
        sio.savemat(self.vector_out+"{}_semantics_vect.mat".format(self.in_directory.split('/')[-2]), mdict={'semantics_vector':self.vect})
        
        print('Semantic vector saved as {}_semantics_vect.mat'.format(self.in_directory.split('/')[-2]))
        
    def extractPositions(self):
        
        self.vector_type = self.vector_type + ['p']
        
        self.vector_out = self.out_directory + 'vector_files' #Directory for vector files
        if not os.path.exists(self.vector_out):
            os.mkdir(self.vector_out)
        self.vector_out = self.vector_out + '/' 
        
        self.pos_vect = np.zeros((len(self.class_dict), len(os.listdir(self.label_out)),int((self.vect.max()*2)+1))) #Initialize position matrix
        
        self.printProgressBar(0, len(os.listdir(self.label_out)), prefix = 'Extracting Positions:', suffix = 'Complete', length = 50)
        for num, file in enumerate(sorted(os.listdir(self.label_out))): #Iterates through all converted CSV files and extracts positional information
            #print("##################################################### \n")
            #print(num+1,"/",len(os.listdir(self.label_out)))
            self.printProgressBar(num + 1, len(os.listdir(self.label_out)), prefix = 'Extracting Positions:', suffix = 'Complete', length = 50)
            filename = os.fsdecode(file)
            if filename.endswith('.csv'):
                df = pd.read_csv(self.label_out+filename)
                
                #Iterates within each label CSV to extract position information, and calculate centroid of bounding boxes
                for idx,row in df.iterrows(): 
                    self.pos_vect[int(row['Class'])][num][0]+=1
                    centroid_x = (row['x start'] + row['x end'])/2
                    centroid_y = (row['y start'] + row['y end'])/2
                    #print(int(((self.pos_vect[int(row['Class'])][num][0])*2)-1)) ###For debugging purposes###
                    #print(int((self.pos_vect[int(row['Class'])][num][0])*2)) ###For debugging purposes###
                    self.pos_vect[int(row['Class'])][num][int(((self.pos_vect[int(row['Class'])][num][0])*2)-1)] = centroid_x
                    self.pos_vect[int(row['Class'])][num][int((self.pos_vect[int(row['Class'])][num][0])*2)] = centroid_y
                    
        sio.savemat(self.vector_out +"{}_positions_vect.mat".format(self.in_directory.split('/')[-2]), mdict={'positions_vector':self.pos_vect})
        print('Positions vector saved as {}_positions_vect.mat'.format(self.in_directory.split('/')[-2]))
    
    def getEuclidean(self, thresh=50): #Thresh only applies if the euclidean distances are being extracted from positional data
        if self.vector_type == ['s']:
            print('Semantic information detected.')
            print('Computing euclidean distances of semantic data.')
            self.semantic_euc = np.array([]) #Initialize vector for semantic euclidean distance
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances:', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1): #Iterate through semantic vector, and perform euclidean distance calculations
                a = self.vect[:,i]
                b = self.vect[:,i+1]
                semDst = distance.euclidean(a,b)
                self.semantic_euc = np.append(self.semantic_euc, semDst)
                #print(semDst)
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances:', suffix = 'Complete', length = 50)
            
            sio.savemat(self.vector_out+"{}_euclidean_semantics.mat".format(self.in_directory.split('/')[-2]), mdict={'euclidean_semantics':self.semantic_euc})
            print('Semantic euclidean distances vector saved as {}_euclidean_semantics.mat'.format(self.in_directory.split('/')[-2]))
            
        elif self.vector_type == ['p']:
            print('Positional information detected.')
            print('Computing euclidean distances of positional data.')
            self.position_euc = np.zeros((len(self.class_dict), len(os.listdir(self.label_out))-1,int(self.vect.max())+1)) #Initialize positional euclidean distance matrix
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances:', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1): #Iterate through frames axis of position vector
                #print('##################################################### \n')
                #print('Frame ',i+1,' / ',len(os.listdir(self.label_out)))
                self.printProgressBar(i+1, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances:', suffix = 'Complete', length = 50)
                for j in range(0, len(self.class_dict)): #Iterate through label classes axis of position vector
                    euctemppositions = []
                    if self.pos_vect[j][i][0] == 0:
                        continue
                    
                    for idx, k in enumerate(self.pos_vect[j][i]): #Iterate through centroids axis of position vector 
                        if idx == 0:
                            self.position_euc[j][i][0] = k
                        elif idx > 0: 
                            if (idx % 2 != 0):
                                next_pos_num = 1 #Initialized at 1, to be incremented in upcoming while-loop
                                euctemp = []
                                if (self.pos_vect[j][i][idx] == 0) and (self.pos_vect[j][i][idx+1] == 0): #Assumption made here is that centroids (if exists) will never be at 0,0.
                                    break
                                    
                                while (self.pos_vect[j][i][idx] != 0) and (self.pos_vect[j][i][idx+1] != 0) and (self.pos_vect[j][i+1][(next_pos_num*2)-1] != 0) and (self.pos_vect[j][i+1][(next_pos_num*2)] != 0):
                                    a = self.pos_vect[j][i][idx:idx+2]
                                    b = self.pos_vect[j][i+1][int((next_pos_num*2)-1):int((next_pos_num*2)+1)]
        
                                    eucpos_dst = distance.euclidean(a,b)
                                    euctemp.append(eucpos_dst)
                                    next_pos_num+=1
                                    if (next_pos_num > self.vect.max()):
                                        break
                                if (self.pos_vect[j][i][idx] != 0) and (self.pos_vect[j][i][idx+1] != 0) and (self.pos_vect[j][i+1][((next_pos_num-1)*2)-1] != 0) and (self.pos_vect[j][i+1][((next_pos_num-1)*2)] != 0):
                                    euctemppositions = euctemppositions + [euctemp] #Constructs a tuple containing every combination of distances
                            
                    while(len(euctemppositions) > 0):
                        idx_min_a = euctemppositions.index(min(euctemppositions)) #index of current frame comparison (corresponds to outer index)
                        idx_min_b = euctemppositions[idx_min_a].index(min(min(euctemppositions))) #index of the next frame comparison (corresponds to inner index)
                        min_dst = min(min(euctemppositions))
                        self.position_euc[j][i][idx_min_a+1] = min_dst #Assign minimum distance to corresponding positional object
                    
                        del euctemppositions[idx_min_a] #deletes index of current frame comparison (outer index) after computation
                            
                        for pos in range(0,len(euctemppositions)): #Iterates through indices of next frame comparisons (inner index) and deletes after computation
                            del euctemppositions[pos][idx_min_b]
        
                        euctemppositions = [position for position in euctemppositions if position]
                    
        
            self.position_euc_percentiles = np.zeros((len(os.listdir(self.label_out))-1))
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Cleaning data:', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1):
                eucclean = []
                #print('##################################################### \n')
                #print('Frame ',i+1,' / ',len(os.listdir(self.label_out))-1)
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Cleaning data:', suffix = 'Complete', length = 50)
                for j in range(0, len(self.class_dict)):
                    for idx, k in enumerate(self.position_euc[j][i]):
                        if idx > 0:
                            if not np.isnan(k):
                                eucclean.append(k)
                
                eucclean = np.trim_zeros(eucclean)   
                if len(eucclean) == 0:
                    self.position_euc_percentiles[i] = 0
                    print("length is zero")
                else:    
                    self.position_euc_percentiles[i] = np.percentile(eucclean,95)
                    
                if np.diff(self.position_euc_percentiles[i-1:i+1]) > thresh: #Specified spike filtering threshold
                    self.position_euc_percentiles[i] = self.position_euc_percentiles[i-1]
                if self.position_euc_percentiles[i] != 0:
                    print("not zero")
                    
            
            sio.savemat(self.vector_out+"{}_euclidean_positions.mat".format(self.in_directory.split('/')[-2]), mdict={'euclidean_positions':self.position_euc_percentiles})
            print('Positional euclidean distances vector saved as {}_euclidean_positions.mat'.format(self.in_directory.split('/')[-2]))
            
        elif ('p' in self.vector_type) and ('s' in self.vector_type):
            print('Semantic and Positional information detected.')
            print('Computing euclidean distances.')
            self.semantic_euc = np.array([]) #Initialize vector for semantic euclidean distance
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances (1/2):', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1): #Iterate through semantic vector, and perform euclidean and cosine distance calculations
                a = self.vect[:,i]
                b = self.vect[:,i+1]
                eucSemDst = distance.euclidean(a,b)
                self.semantic_euc = np.append(self.semantic_euc, eucSemDst)
                #print(eucSemDst)
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances (1/2):', suffix = 'Complete', length = 50)
            
            sio.savemat(self.vector_out+"{}_euclidean_semantics.mat".format(self.in_directory.split('/')[-2]), mdict={'euclidean_semantics':self.semantic_euc})
            print('Semantic euclidean distances vector saved as {}_euclidean_semantics.mat'.format(self.in_directory.split('/')[-2]))
            
            
            self.position_euc = np.zeros((len(self.class_dict), len(os.listdir(self.label_out))-1,int(self.vect.max())+1)) #Initialize positional euclidean distance matrix
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances (2/2):', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1): #Iterate through frames axis of position vector
                #print('##################################################### \n')
                #print('Frame ',i+1,' / ',len(os.listdir(self.label_out)))
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances (2/2):', suffix = 'Complete', length = 50)
                for j in range(0, len(self.class_dict)): #Iterate through label classes axis of position vector
                    euctemppositions = []
                    if self.pos_vect[j][i][0] == 0:
                        continue
                    
                    for idx, k in enumerate(self.pos_vect[j][i]): #Iterate through centroids axis of position vector 
                        if idx == 0:
                            self.position_euc[j][i][0] = k
                        elif idx > 0: 
                            if (idx % 2 != 0):
                                next_pos_num = 1 #Initialized at 1, to be incremented in upcoming while-loop
                                euctemp = []
                                if (self.pos_vect[j][i][idx] == 0) and (self.pos_vect[j][i][idx+1] == 0): #Assumption made here is that centroids (if exists) will never be at 0,0.
                                    break
                                    
                                while (self.pos_vect[j][i][idx] != 0) and (self.pos_vect[j][i][idx+1] != 0) and (self.pos_vect[j][i+1][(next_pos_num*2)-1] != 0) and (self.pos_vect[j][i+1][(next_pos_num*2)] != 0):
                                    a = self.pos_vect[j][i][idx:idx+2]
                                    b = self.pos_vect[j][i+1][int((next_pos_num*2)-1):int((next_pos_num*2)+1)]
        
                                    eucpos_dst = distance.euclidean(a,b)
                                    euctemp.append(eucpos_dst)
                                    next_pos_num+=1
                                    if (next_pos_num > self.vect.max()):
                                        break
                                if (self.pos_vect[j][i][idx] != 0) and (self.pos_vect[j][i][idx+1] != 0) and (self.pos_vect[j][i+1][((next_pos_num-1)*2)-1] != 0) and (self.pos_vect[j][i+1][((next_pos_num-1)*2)] != 0):
                                    euctemppositions = euctemppositions + [euctemp] #Constructs a tuple containing every combination of distances
                            
                    while(len(euctemppositions) > 0):
                        idx_min_a = euctemppositions.index(min(euctemppositions)) #index of current frame comparison (corresponds to outer index)
                        idx_min_b = euctemppositions[idx_min_a].index(min(min(euctemppositions))) #index of the next frame comparison (corresponds to inner index)
                        min_dst = min(min(euctemppositions))
                        self.position_euc[j][i][idx_min_a+1] = min_dst #Assign minimum distance to corresponding positional object
                    
                        del euctemppositions[idx_min_a] #deletes index of current frame comparison (outer index) after computation
                            
                        for pos in range(0,len(euctemppositions)): #Iterates through indices of next frame comparisons (inner index) and deletes after computation
                            del euctemppositions[pos][idx_min_b]
        
                        euctemppositions = [position for position in euctemppositions if position]
                    
        
            self.position_euc_percentiles = np.zeros((len(os.listdir(self.label_out))-1))
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Cleaning data:', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1):
                eucclean = []
                #print('##################################################### \n')
                #print('Frame ',i+1,' / ',len(os.listdir(self.label_out))-1)
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Cleaning data:', suffix = 'Complete', length = 50)
                for j in range(0, len(self.class_dict)):
                    for idx, k in enumerate(self.position_euc[j][i]):
                        if idx > 0:
                            if not np.isnan(k):
                                eucclean.append(k)
                
                eucclean = np.trim_zeros(eucclean)   
                if len(eucclean) == 0:
                    self.position_euc_percentiles[i] = 0
                    #print("length is zero") #For debugging
                else:    
                    self.position_euc_percentiles[i] = np.percentile(eucclean,95)
                    
                if np.diff(self.position_euc_percentiles[i-1:i+1]) > thresh: #Specified spike filtering threshold
                    self.position_euc_percentiles[i] = self.position_euc_percentiles[i-1]
                #if self.position_euc_percentiles[i] != 0:
                    #print("not zero") #For debugging
                    
            
            sio.savemat(self.vector_out+"{}_euclidean_positions.mat".format(self.in_directory.split('/')[-2]), mdict={'euclidean_positions':self.position_euc_percentiles})
            print('Positional euclidean distances vector saved as {}_euclidean_positions.mat'.format(self.in_directory.split('/')[-2]))

    def getCosine(self):
        if self.vector_type == ['s']:
            print('Semantic information detected.')
            print('Computing cosine distances of semantic data.')
            self.semantic_cos = np.array([]) #Initialize vector for semantic cosine distance
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Computing cosine distances:', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1): #Iterate through semantic vector, and perform cosine distance calculations
                a = self.vect[:,i]
                b = self.vect[:,i+1]
                semCosDst = distance.cosine(a,b)
                self.semantic_cos = np.append(self.semantic_cos, semCosDst)
                #print(semCosDst) #For debugging
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Computing cosine distances:', suffix = 'Complete', length = 50)
            
            sio.savemat(self.vector_out+"{}_cosine_semantics.mat".format(self.in_directory.split('/')[-2]), mdict={'cosine_semantics':self.semantic_cos})
            print('Semantic cosine distances vector saved as {}_cosine_semantics.mat'.format(self.in_directory.split('/')[-2]))
            
            
        elif self.vector_type == ['p']:
            print('Positional information detected.')
            print('Computing cosine distances of positional data.')
            self.position_cos = np.zeros((len(self.class_dict), len(os.listdir(self.label_out))-1,int(self.vect.max())+1)) #Initialize positional cosine distance matrix
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Computing cosine distances:', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1): #Iterate through frames axis of position vector
                #print('##################################################### \n')
                #print('Frame ',i+1,' / ',len(os.listdir(self.label_out)))
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Computing cosine distances:', suffix = 'Complete', length = 50)
                
                for j in range(0, len(self.class_dict)): #Iterate through label classes axis of position vector
                    costemppositions = []
                    if self.pos_vect[j][i][0] == 0:
                        continue
                    
                    for idx, k in enumerate(self.pos_vect[j][i]): #Iterate through centroids axis of position vector 
                        if idx == 0:
                            self.position_cos[j][i][0] = k
                        elif idx > 0: 
                            if (idx % 2 != 0):
                                next_pos_num = 1 #Initialized at 1, to be incremented in upcoming while-loop
                                costemp = []
                                if (self.pos_vect[j][i][idx] == 0) and (self.pos_vect[j][i][idx+1] == 0): #Assumption made here is that centroids (if exists) will never be at 0,0.
                                    break
                                    
                                while (self.pos_vect[j][i][idx] != 0) and (self.pos_vect[j][i][idx+1] != 0) and (self.pos_vect[j][i+1][(next_pos_num*2)-1] != 0) and (self.pos_vect[j][i+1][(next_pos_num*2)] != 0):
                                    a = self.pos_vect[j][i][idx:idx+2]
                                    b = self.pos_vect[j][i+1][int((next_pos_num*2)-1):int((next_pos_num*2)+1)]
        
                                    cospos_dst = distance.cosine(a,b)
                                    costemp.append(cospos_dst)
                                    next_pos_num+=1
                                    if (next_pos_num > self.vect.max()):
                                        break
                                if (self.pos_vect[j][i][idx] != 0) and (self.pos_vect[j][i][idx+1] != 0) and (self.pos_vect[j][i+1][((next_pos_num-1)*2)-1] != 0) and (self.pos_vect[j][i+1][((next_pos_num-1)*2)] != 0):
                                    costemppositions = costemppositions + [costemp] #Constructs a tuple containing every combination of distances
                            
                    while(len(costemppositions) > 0):
                        idx_min_a = costemppositions.index(min(costemppositions)) #index of current frame comparison (corresponds to outer index)
                        idx_min_b = costemppositions[idx_min_a].index(min(min(costemppositions))) #index of the next frame comparison (corresponds to inner index)
                        min_dst = min(min(costemppositions))
                        self.position_cos[j][i][idx_min_a+1] = min_dst #Assign minimum distance to corresponding positional object
                    
                        del costemppositions[idx_min_a] #deletes index of current frame comparison (outer index) after computation
                            
                        for pos in range(0,len(costemppositions)): #Iterates through indices of next frame comparisons (inner index) and deletes after computation
                            del costemppositions[pos][idx_min_b]
        
                        costemppositions = [position for position in costemppositions if position]
                    
        
            self.position_cos_percentiles = np.zeros((len(os.listdir(self.label_out))-1))
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Cleaning data:', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1):
                cosclean = []
                #print('##################################################### \n')
                #print('Frame ',i+1,' / ',len(os.listdir(self.label_out))-1)
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Cleaning data:', suffix = 'Complete', length = 50)
                
                for j in range(0, len(self.class_dict)):
                    for idx, k in enumerate(self.position_cos[j][i]):
                        if idx > 0:
                            if not np.isnan(k):
                                cosclean.append(k)
                
                cosclean = np.trim_zeros(cosclean)   
                if len(cosclean) == 0:
                    self.position_cos_percentiles[i] = 0
                    #print("length is zero") for debugging 
                else:    
                    self.position_cos_percentiles[i] = np.percentile(cosclean,95)
                    
            
            sio.savemat(self.vector_out+"{}_cosine_positions.mat".format(self.in_directory.split('/')[-2]), mdict={'cosine_positions':self.position_cos_percentiles}) 
            print('Positional cosine distances vector saved as {}_cosine_positions.mat'.format(self.in_directory.split('/')[-2]))
            
        elif ('p' in self.vector_type) and ('s' in self.vector_type):
        
            print('Semantic and Positional information detected.')
            print('Computing euclidean distances.')
            
            self.semantic_cos = np.array([]) #Initialize vector for semantic cosine distance
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances (1/2):', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1): #Iterate through semantic vector, and perform cosine distance calculations
                a = self.vect[:,i]
                b = self.vect[:,i+1]
                semCosDst = distance.cosine(a,b)
                self.semantic_cos = np.append(self.semantic_cos, semCosDst)
                #print(semCosDst)
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances (1/2):', suffix = 'Complete', length = 50)
            
            sio.savemat(self.vector_out+"{}_cosine_semantics.mat".format(self.in_directory.split('/')[-2]), mdict={'cosine_semantics':self.semantic_cos})
            print('Semantic cosine distances vector saved as {}_cosine_semantics.mat'.format(self.in_directory.split('/')[-2]))
            
            self.position_cos = np.zeros((len(self.class_dict), len(os.listdir(self.label_out))-1,int(self.vect.max())+1)) #Initialize positional cosine distance matrix
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances (2/2):', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1): #Iterate through frames axis of position vector
                #print('##################################################### \n')
                #print('Frame ',i+1,' / ',len(os.listdir(self.label_out)))
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Computing euclidean distances (2/2):', suffix = 'Complete', length = 50)
                
                for j in range(0, len(self.class_dict)): #Iterate through label classes axis of position vector
                    costemppositions = []
                    if self.pos_vect[j][i][0] == 0:
                        continue
                    
                    for idx, k in enumerate(self.pos_vect[j][i]): #Iterate through centroids axis of position vector 
                        if idx == 0:
                            self.position_cos[j][i][0] = k
                        elif idx > 0: 
                            if (idx % 2 != 0):
                                next_pos_num = 1 #Initialized at 1, to be incremented in upcoming while-loop
                                costemp = []
                                if (self.pos_vect[j][i][idx] == 0) and (self.pos_vect[j][i][idx+1] == 0): #Assumption made here is that centroids (if exists) will never be at 0,0.
                                    break
                                    
                                while (self.pos_vect[j][i][idx] != 0) and (self.pos_vect[j][i][idx+1] != 0) and (self.pos_vect[j][i+1][(next_pos_num*2)-1] != 0) and (self.pos_vect[j][i+1][(next_pos_num*2)] != 0):
                                    a = self.pos_vect[j][i][idx:idx+2]
                                    b = self.pos_vect[j][i+1][int((next_pos_num*2)-1):int((next_pos_num*2)+1)]
        
                                    cospos_dst = distance.cosine(a,b)
                                    costemp.append(cospos_dst)
                                    next_pos_num+=1
                                    if (next_pos_num > self.vect.max()):
                                        break
                                if (self.pos_vect[j][i][idx] != 0) and (self.pos_vect[j][i][idx+1] != 0) and (self.pos_vect[j][i+1][((next_pos_num-1)*2)-1] != 0) and (self.pos_vect[j][i+1][((next_pos_num-1)*2)] != 0):
                                    costemppositions = costemppositions + [costemp] #Constructs a tuple containing every combination of distances
                            
                    while(len(costemppositions) > 0):
                        idx_min_a = costemppositions.index(min(costemppositions)) #index of current frame comparison (corresponds to outer index)
                        idx_min_b = costemppositions[idx_min_a].index(min(min(costemppositions))) #index of the next frame comparison (corresponds to inner index)
                        min_dst = min(min(costemppositions))
                        self.position_cos[j][i][idx_min_a+1] = min_dst #Assign minimum distance to corresponding positional object
                    
                        del costemppositions[idx_min_a] #deletes index of current frame comparison (outer index) after computation
                            
                        for pos in range(0,len(costemppositions)): #Iterates through indices of next frame comparisons (inner index) and deletes after computation
                            del costemppositions[pos][idx_min_b]
        
                        costemppositions = [position for position in costemppositions if position]
                    
        
            self.position_cos_percentiles = np.zeros((len(os.listdir(self.label_out))-1))
            
            self.printProgressBar(0, len(os.listdir(self.label_out))-1, prefix = 'Cleaning data:', suffix = 'Complete', length = 50)
            for i in range(0,len(os.listdir(self.label_out))-1):
                cosclean = []
                #print('##################################################### \n')
                #print('Frame ',i+1,' / ',len(os.listdir(self.label_out))-1)
                self.printProgressBar(i + 1, len(os.listdir(self.label_out))-1, prefix = 'Cleaning data:', suffix = 'Complete', length = 50)
                for j in range(0, len(self.class_dict)):
                    for idx, k in enumerate(self.position_cos[j][i]):
                        if idx > 0:
                            if not np.isnan(k):
                                cosclean.append(k)
                
                cosclean = np.trim_zeros(cosclean)   
                if len(cosclean) == 0:
                    self.position_cos_percentiles[i] = 0
                    print("length is zero")
                else:    
                    self.position_cos_percentiles[i] = np.percentile(cosclean,95)
                    
            
            sio.savemat(self.vector_out+"{}_cosine_positions.mat".format(self.in_directory.split('/')[-2]), mdict={'cosine_positions':self.position_cos_percentiles})     
            print('Positional cosine distances vector saved as {}_cosine_positions.mat'.format(self.in_directory.split('/')[-2]))
            
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total: 
            print()


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()   