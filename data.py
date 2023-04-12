# -*- coding: utf-8 -*-
import glob
import cv2
import numpy as np
import keras
# import os
# from os.path import isfile, join

# load all training data into array
def datagenerator(data_dir ='./train/', scale=3):
    crop_size_lr = 32
    crop_size_hr = 32 * scale
    filenames = glob.glob(data_dir+'/*')
    # path = data_dir+'/'
    # files = [f for f in os.listdir(path) if isfile(join(path, f))]
    # files.sort()
    x = []
    y = []
    for i in range(0, len(filenames), 3):
    # for i in range(len(filenames)):
        im1 = cv2.imread(filenames[i]).astype(np.float32) / 255.0
        im2 = cv2.imread(filenames[i+1], 3).astype(np.float32) / 255.0
        im3 = cv2.imread(filenames[i+2], 3).astype(np.float32) / 255.0
        
        imgYCC1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
        imgYCC2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
        imgYCC3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)
        
        width, height,c = im2.shape
        imgYCC = np.empty((width,height,3))
        imgYCC[:,:,0] = imgYCC1[:,:,0]
        imgYCC[:,:,1] = imgYCC2[:,:,0]
        imgYCC[:,:,2] = imgYCC3[:,:,0]
        
        # imgYCC = np.concatenate((imgYCC1[:,:,0], imgYCC2[:,:,0], imgYCC3[:,:,0]), axis=2)
        
        cropped = imgYCC[0:(imgYCC.shape[0] - (imgYCC.shape[0] % scale)),
                  0:(imgYCC.shape[1] - (imgYCC.shape[1] % scale)), :]
        
        lr = cv2.resize(cropped, (int(cropped.shape[1] / scale), int(cropped.shape[0] / scale)),
                        interpolation=cv2.INTER_CUBIC)
        
        hr_y = cropped[:, :, 1]
        lr_y = lr[:, :, :]
        
        numx = int(lr.shape[0] / crop_size_lr)
        numy = int(lr.shape[1] / crop_size_lr)
        for i in range(0, numx):
            startx = i * crop_size_lr
            endx = (i * crop_size_lr) + crop_size_lr
            startx_hr = i * crop_size_hr
            endx_hr = (i * crop_size_hr) + crop_size_hr
            for j in range(0, numy):
                starty = j * crop_size_lr
                endy = (j * crop_size_lr) + crop_size_lr
                starty_hr = j * crop_size_hr
                endy_hr = (j * crop_size_hr) + crop_size_hr

                crop_lr = lr_y[startx:endx, starty:endy, :]
                crop_hr = hr_y[startx_hr:endx_hr, starty_hr:endy_hr]

                hr = crop_hr.reshape((crop_size_hr, crop_size_hr, 1))
                lr = crop_lr.reshape((crop_size_lr, crop_size_lr, 3))
                x.append(lr)
                y.append(hr)
        
        
    data = np.array(y, dtype='float32')
    data_down = np.array(x, dtype='float32')
    return data_down, data


class DataGenerator1(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    

# def train_datagen(data_dir):
#     n_count = 0
#     epoch_num=5
#     batch_size=32
#     m=[]
#     n=[]
#     while(True):
#         if n_count == 0:
#             xs,ys = datagenerator(data_dir)

#             #normalized
#             # xs = xs.astype('float32') / 255.0
#             # ys = ys.astype('float32') / 255.0
#             indices = list(range(xs.shape[0]))
#             n_count = 1
#         for _ in range(epoch_num):
#             #shuffle
#             np.random.shuffle(indices)    
#             for i in range(0, len(indices), batch_size):
#                 batch_x = xs[indices[i:i + batch_size]]
#                 batch_y = ys[indices[i:i + batch_size]]
#                 m.append(batch_x)
#                 n.append(batch_y)
#                 # yield batch_x, batch_y

#     m = np.array(m, dtype='float32')
#     n = np.array(n, dtype='float32')
#     return n,m

# dir_ = './data/test'
# dir_ = 'D:/Thesis/DataSet/Vid4/gt_cit'
# m, n = datagenerator(dir_)
# m, n = train_datagen(dir_)
