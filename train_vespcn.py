# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from tensorflow.compat.v1.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.compat.v1.keras.optimizers import Adam
import data as dg
import utils
import VESPCN2 as v_model
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
# import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='espcn', type=str, help='select the model save address')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
# parser.add_argument('--train_data', default='./data/train', type=str, help='path of train data')
parser.add_argument('--train_data', default='./Data/Train_GT', type=str, help='path of train data')
parser.add_argument('--test_data', default='./Data/Valid_GT', type=str, help='path of test data')
parser.add_argument('--epoch', default=10, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=10, type=int, help='save model at every x epoches')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--scale_factor', default=3, type=int, help='scale_factor')
parser.add_argument('--mc_independent', default='true', 
                    help='Whether to train motion compensation network independent from super resolution network')
args = parser.parse_args()

# save_dir = os.path.join('model',args.model,'saved_weights')
save_dir = './checkpoints'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# set the GPU parameters
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True 
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
K.set_session(sess)

# dynamically adjust the learning rate
def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch <= 100:
        lr = initial_lr
    elif epoch <= 200:
        lr = initial_lr / 10
    else :
        lr = initial_lr / 100    
    return lr

# the generator of train datasets for fit_generator of keras
def train_datagen(epoch_num=5, batch_size=32, data_dir=args.train_data):
    n_count = 0
    # m=[]
    # n=[]
    while(True):
        if n_count == 0:
            xs,ys = dg.datagenerator(data_dir, args.scale_factor)

            #normalized
            # xs = xs.astype('float32') / 255.0
            # ys = ys.astype('float32') / 255.0
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            #shuffle
            np.random.shuffle(indices)    
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i + batch_size]]
                batch_y = ys[indices[i:i + batch_size]]
                yield batch_x, batch_y



if __name__ == '__main__':
    # get the espcn model
    vespcn = v_model.vespcn(args)
    model = vespcn()
    
    #Printing the summary of the model to see that it doesn't contain quant aware layers\
    model.summary()
    
    train_op = optimizer=Adam(args.lr)
    ps = utils.PSNR
    sm = utils.SSIM
    ms = utils.dct_2d
    # ms = utils.DCT2_Loss
    
    model.compile(train_op, loss='mse' ,metrics=[ps,sm]) #,

    # Early stopping for save best weight
    es_cb = EarlyStopping(monitor='val_loss', patience=1000, verbose=1, mode='auto')
    cp_cb = ModelCheckpoint(filepath = os.path.join(save_dir,'model_DCT_middle_{epoch:03d}.h5'), monitor='val_loss',
                                verbose=1, save_best_only=True, mode='auto')


    #set keras callback function to save the model
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model3_con_16fil_{epoch:03d}.h5'),
                 verbose=1, save_weights_only=True,save_freq='epoch', period=args.save_every)

    # set keras callback function to dynamically adjust the learning rate
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # start train
    x_trn = train_datagen(batch_size=args.batch_size, data_dir=args.train_data)
    x_val = train_datagen(batch_size=args.batch_size, data_dir=args.test_data)
    # x_trn, y_trn =  dg.datagenerator(args.train_data, args.scale_factor)
    history = model.fit(x_trn, epochs=args.epoch, verbose=1,
               steps_per_epoch = 300,
               validation_data = x_val,
               validation_steps = 20,
               callbacks=[lr_scheduler,checkpointer,es_cb,cp_cb])


    
# model_json = model.to_json()
# with open(savedir+ "/chk_2ep.json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights(savedir+ '/chk_2ep.h5')



    #  plot
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    acc1 = history.history['SSIM']
    val_acc1 = history.history['val_SSIM']
    acc2 = history.history['PSNR']
    val_acc2 = history.history['val_PSNR']
    epochs = range(len(loss))
    
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure()
    
    
    
    plt.plot(epochs, acc1, label='Training ssim')
    plt.plot(epochs, val_acc1, label='Validation ssim')
    plt.title('Training and validation SSIM')
    plt.legend()
    plt.figure()
    
    plt.plot(epochs, acc2, label='Training psnr')
    plt.plot(epochs, val_acc2, label='Validation psnr')
    plt.title('Training and validation PSNR')
    plt.legend()
    plt.figure()
    
    plt.show()










