# -*- coding: utf-8 -*-
from tensorflow.compat.v1.keras.layers import  Input, Conv2D, Lambda
from tensorflow.compat.v1.keras.models import Model,load_model
import tensorflow.compat.v1 as tf
import numpy as np
import image_warp as wp
# import tensorflow_model_optimization as tfmot
#


class vespcn:
    def __init__(self, args, scale_factor=4, image_channels=1,loader=False, qat=True):
        self.__name__ = 'vespcn'
        self.scale_factor = args.scale_factor
        self.channels = image_channels
        self.loader = loader
        if hasattr(args, 'mc_independent'):
            self._mc_independent = args.mc_independent
        else:
            self._mc_independent = False
        
    def __call__(self):
        # input_image = Input(shape=(20, 30, 3), name='x')
        # input_image = np.ones([1,64,64,3])
        input_image = Input(shape=(64, 64, 3), name='x')
        # hr = np.ones([5,30,60,1])
        # input_image = [lr, hr]
        
        # neighboring_frames = tf.expand_dims(tf.concat([input_image[0][:, :, :, 0], input_image[0][:, :, :, 2]],
        #                                                       axis=0), axis=3)
        
        neighboring_frames = tf.expand_dims(tf.concat([input_image[:, :, :, 0],
                                                        input_image[:, :, :, 2]],
                                                               axis=0), axis=3)
        # m0 = input_image[:, :, :, 0]
        # m1 = input_image[:, :, :, 1]
        # m2 = input_image[:, :, :, 2]
        
        # st0 = tf.stack([m1, m0] ,axis=3)
        # st1 = tf.stack([m1, m2] ,axis=3)
        # lr_input = tf.concat([st0, st1], axis=0)
        lr_input = tf.concat([tf.stack([input_image[:, :, :, 1], input_image[:, :, :, 0]], axis=3)
                              ,tf.stack([input_image[:, :, :, 1], input_image[:, :, :, 2]], axis=3)], axis=0)
        
        print('************** CoarseFlow net **************')
        # CoarseFlow net
        coarse_net = Conv2D(24, 5, strides=2, kernel_initializer = 'he_normal', padding='same',
                                    activation=tf.nn.relu, name="coarse1")(lr_input)
        coarse_net = Conv2D(24, 3, strides=1, kernel_initializer = 'he_normal', padding='same',
                                   activation=tf.nn.relu, name="coarse2")(coarse_net)
        coarse_net = Conv2D(24, 5, strides=2, kernel_initializer = 'he_normal', padding='same',
                                   activation=tf.nn.relu, name="coarse3")(coarse_net)
        coarse_net = Conv2D(24, 3, strides=1, kernel_initializer = 'he_normal', padding='same',
                                   activation=tf.nn.relu, name="coarse4")(coarse_net)
        coarse_net = Conv2D(32, 3, strides=1, kernel_initializer = 'he_normal', padding='same',
                                   activation=tf.nn.tanh, name="coarse5")(coarse_net)
       
        coarse_flow = 36.0 * tf.depth_to_space(coarse_net, 4)
        
        warped_frames1 = wp.image_warp(neighboring_frames, coarse_flow)
        
        # warped_frames1 = wp.dense_image_warp(neighboring_frames, coarse_flow)
        
        
        ff_input = tf.concat([lr_input, coarse_flow, warped_frames1], axis=3)
        
        print('************** FineFlow net **************')
        
        # FineFlow Net
        fine_net = Conv2D(24, 5, strides=2, kernel_initializer = 'he_normal', padding='same',
                                    activation=tf.nn.relu, name="fine1")(ff_input)
        fine_net = Conv2D(24, 3, strides=1, kernel_initializer = 'he_normal', padding='same',
                                   activation=tf.nn.relu, name="fine2")(fine_net)
        fine_net = Conv2D(24, 3, strides=1, kernel_initializer = 'he_normal', padding='same',
                                   activation=tf.nn.relu, name="fine3")(fine_net)
        fine_net = Conv2D(24, 3, strides=1, kernel_initializer = 'he_normal', padding='same',
                                   activation=tf.nn.relu, name="fine4")(fine_net)
        fine_net = Conv2D(8, 3, strides=1, kernel_initializer = 'he_normal', padding='same',
                                   activation=tf.nn.tanh, name="fine5")(fine_net)
        fine_flow = 36.0 * tf.depth_to_space(fine_net, 2)
        flow = coarse_flow + fine_flow
        
        warped_frames = wp.image_warp(neighboring_frames, flow)
        # warped_frames = wp.dense_image_warp(neighboring_frames, flow)
        
        
        # if self._mc_independent:
        # sr_input1 = tf.concat([tf.stop_gradient(warped_frames[:tf.shape(input_image)[0]]),
        #                       input_image[:, :, :, 1:2],
        #                       tf.stop_gradient(warped_frames[tf.shape(input_image)[0]:])], axis=3)
        # # else:
        sr_input1 = tf.concat([warped_frames[:tf.shape(input_image)[0]],
                                  input_image[:, :, :, 1:2],
                                      warped_frames[tf.shape(input_image)[0]:]], axis=3)
        
        sr_input = tf.pad(sr_input1, [[0, 0], [5, 5], [5, 5], [0, 0]], 'SYMMETRIC')
        
        print('************** SR net **************')
        # net = Conv2D(64, 5, kernel_initializer='glorot_uniform', padding='valid', activation=tf.nn.relu,name="conv1")(sr_input)
        # net = Conv2D(32, 3, kernel_initializer='glorot_uniform', padding='valid', activation=tf.nn.relu,name="conv2")(net)
        # net = Conv2D(self.scale_factor**2*self.channels, 3, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv3")(net)
        
        net = Conv2D(64, 5, kernel_initializer = 'he_normal', padding='valid',
                                    activation=tf.nn.relu, name="conv1")(sr_input)
        net = Conv2D(32, 3, kernel_initializer = 'he_normal', padding='valid',
                                    activation=tf.nn.relu, name="conv2")(net)
        net = Conv2D(32, 3, kernel_initializer = 'he_normal', padding='valid',
                                    activation=tf.nn.relu, name="conv3")(net)
        #net = Conv2D(24, 3, kernel_initializer = 'he_normal', padding='valid',
        #                            activation=tf.nn.relu, name="conv4")(net)
        net = Conv2D(self.scale_factor ** 2, 3,kernel_initializer = 'he_normal', padding='valid',
                                    activation=None, name='conv5')(net)
        # predicted_batch = tf.depth_to_space(net, self._scale_factor, name='prediction')
        
        predicted = tf.depth_to_space(net, self.scale_factor,name="prediction")
        
        model = Model(inputs=input_image, outputs=predicted)
        model.summary()
        return model

class espcn:
    def __init__(self, scale_factor=4, image_channels=1,loader=False, qat=True):
        self.__name__ = 'espcn'
        self.scale_factor = scale_factor
        self.channels = image_channels
        self.loader = loader

    # upsampling the resolution of image
    def sub_pixel(self, x):
        return tf.compat.v1.depth_to_space(x, self.scale_factor,name="prediction")
        
    # building the espcn network
    def __call__(self):
        if self.loader is True:
            input_image = Input(shape=(240, 432, self.channels), name='x')
        else:    
            input_image = Input(shape=(17, 17, 3), name='x')
        x = Conv2D(64, 5, kernel_initializer='glorot_uniform', padding='same', activation=tf.nn.relu,name="conv1")(input_image)
        x = Conv2D(32, 3, kernel_initializer='glorot_uniform', padding='same', activation=tf.nn.relu,name="conv2")(x)
        # x = Conv2D(16, 1, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv0")(x)
        x = Conv2D(self.scale_factor**2*self.channels, 3, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv3")(x)
        if self.scale_factor > 1:
            y = self.sub_pixel(x)
        model = Model(inputs=input_image, outputs=y)
        # model.summary()
        return model
