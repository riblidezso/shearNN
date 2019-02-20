#!/usr/bin/env python
# coding: utf-8
# *Author: Dezso Ribli*


"""
Multi functional CNN training script.
"""


###############################################################################
# modules
import numpy as np
import pickle
import argparse
from functools import partial
import os

###############################################################################
# args
parser = argparse.ArgumentParser()

# general setup:
parser.add_argument("--survey", type=str, choices=['des'],
                    default = 'des', help="Survey data used\n")

# targets to predict
parser.add_argument('--targets', nargs='+', help='Targets to predict\n', 
                    required=True)
# drop lowq data
parser.add_argument("--scale-targets", help="Scale targets\n", 
                    action="store_true")

# input details
parser.add_argument("--im-size", type=int, choices=[50], default=50,
                    help="Image size in pixels")
# image colors
parser.add_argument("--colors", type=str, default = 'ygriz',
                    choices=['ygriz'], help="Colors used\n")
# drop lowq data
parser.add_argument("--drop-lowq", help="Drop low quality ground truth \n", 
                    action="store_true")

# cnn params
parser.add_argument("--n-filters", type=int, choices=[16,32,64], default = 16,
                    help="Number of filters in the first conv layer\n")
parser.add_argument("--l2reg", type=float, default = 5e-5,
                    help="L2 regularization coefficient")
parser.add_argument("--cnn-weights", type=str, 
                    help="Load pretrained weights from file")

# testing setup
parser.add_argument("--split", type=str, choices=['radec','random'],
                    default = 'random', help="Train/test split\n")
parser.add_argument("--test-size", type=float,
                    default = 0.1, help="Test size for random split\n")
parser.add_argument("--train-size", type=float,
                    default = 1.0, help="Train size\n")

# data augmentation
parser.add_argument("--augment-train", help="Augment train images\n", 
                    action="store_true")

# training params
parser.add_argument("--gpu", type=str, help="GPU id\n")
parser.add_argument("--reserve-vram", type=float, default=0.9,
                    help="Ratio of memory reserved on gpu.\n")
parser.add_argument("--batch-size", type=int, default = 128,
                    help="Mini-batch size\n")
parser.add_argument("--base-lr", type=float, default = 0.005,
                    help="Base learning rate\n")
parser.add_argument("--n-epochs", type=int, default = 30,
                    help="Number of epochs to train.\n")
parser.add_argument('--epochs-drop', nargs='+', required=True, type=int,
                    help='Drop LR 10-fold after epochs\n')
parser.add_argument("--steps-per-epoch", type=int, default = None,
                    help="Number of updates in an epoch, one epoch is the \
                         full training set if not set.\n")      
parser.add_argument("--loss", type=str, choices=['mean_absolute_error',
                    'mean_squared_error'],
                    default = 'mean_absolute_error',
                    help="Training loss function \n")
parser.add_argument("--seed", type=int, default = 0,
                    help="Random seed used for data shuffling and random\
                    train-test splits.\n")
parser.add_argument("--only-predict", 
                    help="Skip training, only predict if model was saved.\n",
                    action="store_true")

# Training from storage, not used now!!!
parser.add_argument("--from-files", 
                    help="Read maps from files when generator datat instead \
                    of reading from memory\n", action="store_true")
parser.add_argument("--dir", type=str, default = '../../data/',\
                    help="Base directory for data if reading from files\n")

args = parser.parse_args()
print '\n\n',args,'\n\n'  # report args collected

###############################################################################
# set gpu id, and only 90% to keep the N-body simulation alive
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.reserve_vram
config.gpu_options.visible_device_list = args.gpu
set_session(tf.Session(config=config))

#keras only after gpu ID and memory usage is set
from keras.callbacks import LearningRateScheduler
from keras import optimizers

# custom stuff 
from utils import load_training_data, step_decay, DataGenerator
from utils import predict_on_generator
from nn import mycnn

###############################################################################
# assign a simpler run-name to find data product file names
# more detailed args namespace is saved with preidctions too
RUN_NAME = 'pix50_'+ '-'.join(args.targets) + '_'+args.split
RUN_NAME += '_' + args.colors + '_'+args.survey + '_gpu'+args.gpu
RUN_NAME += '_%.2f' % args.train_size
RUN_NAME +=  '_%.2f' % args.test_size +'_seed%s'%args.seed+'_loss-'+ args.loss
RUN_NAME += '_nfilt-%d' % args.n_filters
if args.augment_train:
    RUN_NAME+='_augtrain'
if args.drop_lowq:
    RUN_NAME += '_droplowq'
print 'Run name', RUN_NAME

###############################################################################
# create CNN model
model = mycnn(imsize=args.im_size, n_target = len(args.targets), 
              reg=args.l2reg, nf=args.n_filters, n_channels=len(args.colors),
              gpu=args.gpu)
# simple SGD (later there will be step decay added to the train function)
sgd = optimizers.SGD(lr=args.base_lr, decay=0, momentum=0.9, nesterov=True)
# compile
model.compile(loss=args.loss, optimizer=sgd, metrics=[args.loss])
print model.summary()  # print a summary

if args.cnn_weights is not None:  # load model weights if specified
    model.load_weights(args.cnn_weights, by_name=True)
    
###############################################################################
# load / prepare data
data = load_training_data(args.survey, args.targets, args.colors, 
                          args.split, args.test_size, args.train_size,
                          args.aux_inputs,  seed=args.seed,
                          drop_lowq=args.drop_lowq)
X_train, X_test, y_train, y_test = data

if args.scale_targets:  # scale targets to std == 1
    scales = y_train.std(axis=0)
    y_train /= scales
    y_test /= scales

# create data generators for on the fly augmentations
dg_train = DataGenerator(X_train, y_train,
                         batch_size=args.batch_size, seed=args.seed,
                         augment=args.augment_train, im_size=args.im_size, 
                         n_channels = len(args.colors),
                         y_shape = (len(args.targets),))
dg_test = DataGenerator(X_test, y_test,
                        batch_size=args.batch_size, im_size=args.im_size, 
                        y_shape = (len(args.targets),), shuffle=False,
                        n_channels = len(args.colors))

###############################################################################
# train and save model
n_steps = args.steps_per_epoch if args.steps_per_epoch else dg_train.n_steps
sdecay = partial(step_decay,base_lr=args.base_lr,epochs_drop=args.epochs_drop)
if not args.only_predict:
    model.fit_generator(dg_train,  nb_epoch=args.n_epochs,
                        steps_per_epoch=n_steps, 
                        validation_data=dg_test, 
                        validation_steps=dg_test.n_steps,
                        callbacks=[LearningRateScheduler(sdecay)], verbose=2)
    model.save_weights('results/cnn_model_'+RUN_NAME+'.p')  # save the model
else:
    model.load_weights('results/cnn_model_'+RUN_NAME+'.p')  # load the model

###############################################################################
# make predictions on test set
y_true_test, p_test = predict_on_generator(model, dg_test)
# get rid of extra preds
y_true_test, p_test = y_true_test[:len(y_test)], p_test[:len(y_test)]

if args.scale_targets:  # scale back targets to original scale
    y_true_test *= scales
    p_test *= scales

# save predicitons
with open('results/cnn_predictions_' + RUN_NAME + '.pkl', 'wb') as fh:
    pickle.dump((y_true_test, p_test, args),fh)
    
###############################################################################
print
print '-----------------------------------'
print 'Finished training and evaluation...'
print args.targets
print 'MAE:', np.abs(y_true_test-p_test).mean(axis=0) 
print 
print
