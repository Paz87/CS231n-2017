# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 01:55:56 2017

@author: pazilan
"""
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import TensorFlowLayers
from TensorFlowLayers import *
import importlib
from importlib import reload


def my_model(X,y,is_training):
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 1024])
    b1 = tf.get_variable("b1", shape=[1024])
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])

    # define our graph
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True, 
                                          is_training=is_training,
                                          scope='bn')
    h3 = tf.contrib.layers.max_pool2d(h2, [2,2])
    h3_flat = tf.reshape(h3,[-1,5408])
    a4 = tf.matmul(h3_flat,W1) + b1
    h4 = tf.nn.relu(a4)
    y_out = tf.matmul(h4,W2) + b2
    return y_out