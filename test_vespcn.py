# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import cv2
from keras.models import load_model
from skimage.io import imsave
import tensorflow.compat.v1 as tf
import VESPCN as v_model
import utils
import datatest as dg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='D:/Thesis/Projects/Video/ESPCN/espcn-keras_3framesIN/data/city', type=str, help='directory of test dataset')
    parser.add_argument('--model_dir', default='./checkpoints', type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_vespcn_scale3_100.h5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results/cit', type=str, help='directory of results')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
    parser.add_argument('--scale_factor', default=3, type=int, help='scale_factor')
    return parser.parse_args()

args = parse_args()
# definition the sub_pixel function because we use Lambda in the model
scale_factor = args.scale_factor
def sub_pixel(x):
    return tf.depth_to_space(x, scale_factor)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # load the model
    # if scale_factor > 1:
    #     model = load_model(os.path.join(args.model_dir, args.model_name),custom_objects={"sub_pixel": sub_pixel}, compile=False)
    # else:
    #     model = load_model(os.path.join(args.model_dir, args.model_name), compile=False)
    i = 0
    
    inputs , CrCb = dg.prepareData(args.set_dir,scale_factor)
    
    
    vespcn = v_model.vespcn(args)
    model = vespcn()
    # model.summary()
    
    # load weights into new model
    pathweight = os.path.join(args.model_dir, args.model_name)
    model.load_weights(pathweight)
    
    print("Loaded model from disk")
    
    model.summary()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    
    
    # import LR_Y and CrCb bicubic 
    
    
    for i in range(len(inputs)):
        # print(file_name)
                
        imgY = inputs[i]
        LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 3)


        # run the model
        output = model.predict(LR_input_)
        # Y = np.squeeze(output, 0)
                
        Y = output[0]
        CrCbch = CrCb[i]
        
        HR_image_YCrCb = np.concatenate((Y, CrCbch), axis=2)
        HR_image = ((cv2.cvtColor(HR_image_YCrCb, cv2.COLOR_YCrCb2BGR)) * 255.0).clip(min=0, max=255)
        HR_image = (HR_image).astype(np.uint8)
        
        # save the super resolution image
        if(i<9):
            result = os.path.join(args.result_dir, 'rec_Frame 00'+str(i+1)+'.png')
        else:
            result = os.path.join(args.result_dir, 'rec_Frame 0'+str(i+1)+'.png')
        imsave(result, HR_image)
        
        
        
