# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:58:24 2017

@author: pazilan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:55:59 2017

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


convLayersDepths = [3,32,64,64,128,128,128] # vgg 16 - convLayersDepths = [3,64,128,256,512,512]
fcLayersDepths = [1024,10]  # vgg 16 - fcLayersDepths = [4096,4096,1000,10] 
convLayersNum = 5  # For now assuming conv layers are the first layers and then the FC layers and no interleaving
fcLayerNum = 2
convFiltSize = [3,3,3,3,3,3]# vgg 16 - convFiltSize = [3,3,3,3]# For now only square filters allowed
numOfDupConvLayers = [4,4,4,4,4,4] # vgg 16 - numOfDupConvLayers = [2,2,3,3,3] # Each conv layer is actually several layers in a row
keep_probs = [1.0,0.6] # First element - keep prob for conv layers and second for FC layers
orgH = X_train.shape[1]
orgW = X_train.shape[2]

def my_model(X,y,is_training):
    currInput = X
    currH = shape(X)[1]
    currW = shape(X)[2]
    inDepth = convLayersDepths[0]
    for i in range(convLayersNum):
        #print(i)
        for j in range(numOfDupConvLayers[i]-1):
            with tf.variable_scope('conv_'+str(i)+str(j)):
                #print('j='+str(j)+"inDepth = "+str(inDepth))
                currInput = ConvLayer(currInput, convFiltSize[i],currH,currW,inDepth, convLayersDepths[i+1],False,True,is_training,keep_prob = keep_probs[0], activation='relu').output()
                inDepth = convLayersDepths[i+1]
        with tf.variable_scope('conv_'+str(i)+str(numOfDupConvLayers[i]-1)):
            currInput = ConvLayer(currInput, convFiltSize[i],currH,currW,inDepth, convLayersDepths[i+1],True,True,is_training,keep_prob = keep_probs[0], activation='relu').output()
            currH = int(currH/2) #curr max pool is only 2x2
            currW = int(currW/2)
    fcInputSize = int((currH*currW*inDepth))
    for i in range(fcLayerNum-1):
        with tf.variable_scope('fc'+str(i)):
#            print("input - " +str(fcInputSize) +" output - " + str(fcLayersDepths[i]))
            currInput = FCLayer(currInput,fcInputSize,fcLayersDepths[i],is_training,keep_prob = keep_probs[1],activation='relu').output()
            fcInputSize = fcLayersDepths[i]
    with tf.variable_scope('fc'+str(fcLayerNum-1)):
 #       print("input - " +str(fcInputSize) +" output - " + str(fcLayersDepths[fcLayerNum-1]))
        currInput = FCLayer(currInput,fcInputSize,fcLayersDepths[fcLayerNum-1],is_training,keep_prob = 1.0,activation=None).output()
    y_out=currInput
    #print(shape(y_out))
    return y_out

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X,y,is_training)
total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.train.AdamOptimizer(1e-4) # select optimizer and set learning rate 1e-4

pass

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)