# -*- coding: utf-8 -*-
import glob
import cv2
import numpy as np
import os
from os.path import isfile, join

# load all data
def LoadData(data_dir, scale):
    
    crcb=[]
    lr_input=[]
    
    filenames = glob.glob(data_dir+'/*')
    for i in range(0, len(filenames)):
        img = cv2.imread(filenames[i]).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width = img.shape[0]
        height = img.shape[1]
        
        cropped = img[0:(width - (width % scale)), 0:(height - (height % scale)), :]#.astype(np.float32)
        imglr = cv2.resize(cropped, None, fx=1. / scale, fy=1. / scale, interpolation=cv2.INTER_CUBIC)
        floatimg = imglr.astype(np.float32) / 255.0   #BGR
        
        
        # Convert to YCrCb color space
        imgYCrCb = cv2.cvtColor(floatimg, cv2.COLOR_BGR2YCrCb)
        imgY = imgYCrCb[:, :, 0]
        # LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 1)
        
        
        Cr = np.expand_dims(cv2.resize(imgYCrCb[:, :, 1], None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_CUBIC), axis=2)
        Cb = np.expand_dims(cv2.resize(imgYCrCb[:, :, 2], None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_CUBIC),axis=2)
        
        imgCrCb = np.concatenate((Cr, Cb), axis=2)
        
        
        lr_input.append(imgY)
        crcb.append(imgCrCb)
        
    lr_input = np.array(lr_input, dtype='float32')
    CrCbch = np.array(crcb, dtype='float32')
    return lr_input, CrCbch
    
    


def prepareData(data_dir, scale):
    
    lr_images , CrCbch = LoadData(data_dir, scale)
    
    
    nframes = 3
    center = nframes//2
    lr_input = []
    for i in range(len(lr_images)):
        frames = np.empty((lr_images.shape[1],lr_images.shape[2],3))
        next_frames = lr_images[i:i+center+1]
        if i<center:
            prev_frames = lr_images[:i]
        else:
            prev_frames = lr_images[i-center:i]
            
        to_fill = nframes - next_frames.shape[0] - prev_frames.shape[0]
        if to_fill:
          if len(prev_frames) and i<nframes:
              pad_x = np.repeat(prev_frames[0][None], to_fill, axis=0)
              frames[:,:,0]=pad_x[0,:,:]
              frames[:,:,1]=prev_frames[0,:,:]
              frames[:,:,2]=next_frames[0,:,:]
          else:
              if i>nframes:
                pad_x = np.repeat(next_frames[-1][None], to_fill, axis=0)
                frames[:,:,0]=prev_frames[0,:,:]
                frames[:,:,1]=next_frames[0,:,:]
                frames[:,:,2]=pad_x[0,:,:]
              else:
                pad_x = np.repeat(next_frames[0][None], to_fill, axis=0)
                frames[:,:,0]=pad_x[0,:,:]
                frames[:,:,1]=next_frames[0,:,:]
                frames[:,:,2]=next_frames[1,:,:]
        else:
            frames[:,:,0]=prev_frames[0,:,:]
            frames[:,:,1]=next_frames[0,:,:]
            frames[:,:,2]=next_frames[1,:,:]
        
        lr_input.append(frames)
      
    lr_input = np.asarray(lr_input, dtype='float32')
    
    return lr_input, CrCbch
    


# dir_ = './data/calendar'
# dir_ = 'D:/Thesis/DataSet/sr_vimeo_3Frames/Train_GT'
# m, n = prepareData(dir_, 3)
