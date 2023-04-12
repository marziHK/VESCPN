# -*- coding: utf-8 -*-
import glob
import cv2
import numpy as np
import math
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
import Gettables as tb


def dct_2d(img):
    X1_t = tf.transpose(img, perm=[0, 1, 3, 2])
    X1 = tf.signal.dct(X1_t, type=2, norm='ortho')
    X2_t = tf.transpose(X1, perm=[0, 1, 3, 2])
    X2 = tf.signal.dct(X2_t, type=2, norm='ortho')
    return X2_t

def DCT2_Loss(y_true, y_pred):
    # for scale2 size table=64
    Quant_Mat1 = tb.GetTable(64)
    Quant_Mat = Quant_Mat1 / 2
    # DCT
    dct_true = dct_2d(y_true)
    dct_pre = dct_2d(y_pred)
    
    # Quantization
    qt = np.expand_dims(Quant_Mat ,axis=-1) 
    #print(' ************ qt ************', qt.shape)
    qt_true = dct_true/qt
    qt_pred = dct_pre/qt
    #print(' ************ qt_true ************', qt_true)
    #print(' ************ qt_pred ************', qt_pred)
    # mse
    mse = K.mean(K.square(qt_true - qt_pred))
    return mse


def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))



def PSNR(im_gt, im_pre):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    # y_pred = im_pre[:,:,:,1]
    # y_true = im_gt[:,:,:,0]
    return 10.0 * K.log(1.0 / (K.mean(K.square(im_pre - im_gt)))) / K.log(10.0)


def Mse1(y_true, y_pred):
  # pred = y_pred[:,:,:,1]
  # gt = y_true[:,:,:,0]
  ms = K.mean(K.square(y_true - y_pred))
  return ms