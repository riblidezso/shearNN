#!/usr/bin/env python
# coding: utf-8
# *Author: Dezso Ribli*

import numpy as np
import os
import math
import pandas as pd
from sklearn.model_selection import train_test_split

def step_decay(epoch, base_lr, epochs_drop, drop=0.1):
    """Helper for step learning rate decay."""
    lrate = base_lr
    for epoch_drop in epochs_drop:
        lrate *= math.pow(drop,math.floor(epoch/epoch_drop))
        return lrate


def load_training_data(survey='des', targets=['e1','e2'], colors='ygriz', 
                       split='random', test_size=0.1, train_size=1.0,
                       normalize=True, seed=0, drop_lowq=True):
    """Load training data and split naively."""
    # load X,y
    if survey == 'des':
        fn='../data/des/dr1/cutouts/des_cutouts50_cfhtolap_%s.npy'%colors
        arr = np.load(fn)
        df = pd.read_csv('../data/cfhtlens/train_desolap_eq_metacal.csv')
        if drop_lowq:  # drop low quality CFHTLenS observations
            idx = df['weight'] > 14
            arr = arr[idx]
            df = df[idx].reset_index(drop=True)
        # remove some observations with -9999 psf ellip (only 2297 removed)
        idx = df['psf_e1'] !=- 9999
        arr = arr[idx]
        df = df[idx].reset_index(drop=True)

    if normalize:  
        for i in range(arr.shape[-1]):
            m = arr[:10000,:,:,i].mean()
            s = arr[:10000,:,:,i].std()
            arr[...,i] = (arr[...,i] - m)/s
            
    # do the train test split
    labels = df[targets].values
    if split == 'random':
        X_train, X_test, y_train, y_test = train_test_split(arr, labels, 
            test_size = test_size, random_state=seed)
    else:
        assert survey=='des'  # only meaningful here now!
        if split == 'radec':
            ralim = np.percentile(df['ra_des'], 100*(1-test_size))
            idx_train = df['ra_des'] < ralim
        X_train, X_test = arr[idx_train], arr[~idx_train] 
        y_train, y_test = labels[idx_train], labels[~idx_train]
            
    # reduce train size
    rng = np.random.RandomState(seed)
    i=rng.permutation(np.arange(len(X_train)))[:int(train_size*len(X_train))]
    X_train,  y_train  = X_train[i],  y_train[i]
    
    return X_train, X_test, y_train, y_test

    
class DataGenerator():
    """
    Data generator.

    Generates minibatches of data and labels.
    Usage:
    from imgen import ImageGenerator
    g = DataGenerator(data, labels)
    """
    def __init__(self, x, y, batch_size=1, shuffle=True, seed=0,
                 im_size = 50, y_shape = (2,), augment = False, n_channels=5):
        """Initialize data generator."""
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.x_shape, self.y_shape = (im_size, im_size, n_channels), y_shape
            
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        
        self.formulation = formulation
        self.nclass = nclass
        assert x.shape[1] == x.shape[2]  # rectangular!!!
        
        self.n_data = len(x)
        self.n_steps = len(x)//batch_size  +  (len(x) % batch_size > 0)
        self.i = 0
        self.reset_indices_and_reshuffle(force=True)
        
        
    def reset_indices_and_reshuffle(self, force=False):
        """Reset indices and reshuffle images when needed."""
        if self.i == self.n_data or force:
            if self.shuffle:
                self.index = self.rng.permutation(self.n_data)
            else:
                self.index = np.arange(self.n_data)
            self.i = 0
            
                
    def next(self):
        """Get next batch of images."""
        x = np.zeros((self.batch_size,)+self.x_shape)
        y = np.zeros((self.batch_size,)+self.y_shape)
        for i in range(self.batch_size):
                    x[i], y[i] = self.next_one()        
        if self.formulation == 'classification':
            y = np.round((self.nclass-1)/2 * (y+1))
            y = [y[:,0], y[:,1]]
        return x,y
    
    
    def next_one(self):
        """Get next 1 image."""
        # reset index, reshuffle if necessary
        self.reset_indices_and_reshuffle()  
        # get next x
        x = np.array(self.x[self.index[self.i]],copy=True)
        y = np.array(self.y[self.index[self.i]],copy=True)
        x, y[0], y[1] = self.process_image(x, e1=y[0], e2=y[1]) # note e1 e2!       
        self.i += 1  # increment counter
        return x, y
    
    
    def process_image(self, x_in, e1, e2):
        """Process data."""
        x = np.array(x_in,copy=True)     
        if self.augment:  # flip and transpose
            # this is not correct now, labels change too!!!
            x, e1, e2 = aug_im(x, e1, e2, self.rng.rand()>0.5, 
                                self.rng.rand()>0.5, self.rng.rand()>0.5)            
        return  x, e1, e2
    
    
def predict_on_generator(model, datagen, augment, formulation='regression'):
    """Predict on data generator with augmentation."""        
    datagen.reset_indices_and_reshuffle(force=True)
    y_true, y_pred = [],[]
    for i in range(datagen.n_steps):
        xi0,yi = datagen.next()
        y_true.append(yi)
        y_pred.append(model.predict_on_batch(xi0))   
    y_true, y_pred = np.vstack(y_true), np.vstack(y_pred)
    return y_true, y_pred


def aug_im(im, e1, e2, fliplr=0, flipud=0, T=0):
    """Augment images with flips and transposition."""
    im = np.array(im, copy=True)
    # calculate eps and theta
    eps = (e1**2 + e2**2)**0.5
    theta =  np.arctan2(e2, e1)/2
    
    if fliplr:  # flip left right
        im = np.fliplr(im)
        # adjust ell
        theta = -theta
        e1, e2 = eps * np.cos(2*theta), eps * np.sin(2*theta)
    if flipud:  # flip up down
        im = np.flipud(im)
        # adjust ell
        theta = np.pi - theta  # but the same as -theta...
        e1, e2 = eps * np.cos(2*theta), eps * np.sin(2*theta)
    if T:  # transpose
        for i in xrange(im.shape[-1]):
            im[:,:,i] = im[:,:,i].T
        # adjust ell
        x,y =  np.sin(theta), np.cos(theta)
        theta = np.arctan2(y, x)
        e1, e2 = eps * np.cos(2*theta), eps * np.sin(2*theta)
    return im, e1, e2