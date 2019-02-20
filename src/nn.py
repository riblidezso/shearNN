#!/usr/bin/env python

import keras.models as km
import keras.layers as kl
import keras.regularizers as kr
import keras.backend as K
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model


def mycnn(imsize, n_target ,n_channels=5, nf=32, reg = 5e-5,
          padding='same', gpu='0'):
    """Return 50 pixel CNN."""
    # conv block
    inp = kl.Input((imsize, imsize, n_channels))
    x = kl.Conv2D(nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(inp)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.Conv2D(nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.MaxPooling2D(strides=(2,2))(x)
    
    # conv block
    x = kl.Conv2D(2*nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.Conv2D(2*nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.MaxPooling2D(strides=(2,2))(x)

    # conv block
    x = kl.Conv2D(4*nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.Conv2D(2*nf, (1, 1), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.Conv2D(4*nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.MaxPooling2D(strides=(2,2))(x)

    # conv block
    x = kl.Conv2D(8*nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.Conv2D(4*nf, (1, 1), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.Conv2D(8*nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.MaxPooling2D(strides=(2,2))(x)
    
    # conv block
    x = kl.Conv2D(16*nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.Conv2D(8*nf, (1, 1), padding=padding,  
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    x = kl.Conv2D(16*nf, (3, 3), padding=padding, 
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation('relu')(kl.BatchNormalization()(x))
    
    #  end of conv
    x = kl.GlobalAveragePooling2D()(x)    
    x = kl.Dense(n_target, name = 'final_dense_n%d_ngpu%d' % (
                n_target, len(gpu.split(','))))(x)  

    model = km.Model(inputs=inp, outputs=x)  # make model

    if len(gpu.split(','))>1:  # multi gpu model
        model = multi_gpu_model(model, gpus=len(gpu.split(',')))
        
    return model

