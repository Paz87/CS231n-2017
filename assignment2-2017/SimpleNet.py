# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:00:10 2017

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



convLayersDepths = [3,64,128,128,128,128,128,128,128,128] 
useMaxPooling = [False,True,False,True,True,False,False,True,True]
fcLayersDepths = [10]
convLayersNum = len(convLayersDepths)-1  # For now assuming conv layers are the first layers and then the FC layers and no interleaving
fcLayerNum = len(fcLayersDepths)
convFiltSize = [3,3,3,3,3,3,1,1,3]
numOfDupConvLayers = [1,3,2,1,2,1,1,1,1] 
keep_probs = [0.7] # First element - keep prob for conv layers and second for FC layers
orgH = 32
orgW = 32

def my_model(X,y,is_training):
    currInput = X
    currH = shape(X)[1]
    currW = shape(X)[2]
    inDepth = convLayersDepths[0]
    for i in range(convLayersNum):
        #print(i)
        for j in range(numOfDupConvLayers[i]-1):
            with tf.variable_scope('conv_'+str(i)+str(j)):
#                print('j='+str(j)+"inDepth = "+str(inDepth))
                currInput = ConvLayer(currInput, convFiltSize[i],currH,currW,inDepth, convLayersDepths[i+1],False,True,is_training,keep_prob = keep_probs[0], activation='relu').output()
                inDepth = convLayersDepths[i+1]
        with tf.variable_scope('conv_'+str(i)+str(numOfDupConvLayers[i]-1)):
            currInput = ConvLayer(currInput, convFiltSize[i],currH,currW,inDepth, convLayersDepths[i+1],useMaxPooling[i],False,is_training,keep_prob = keep_probs[0], activation='relu').output()
            if useMaxPooling[i] : 
                currH = int(currH/2) #curr max pool is only 2x2
                currW = int(currW/2)
#            print('CurrH - ' + str(currH))
#            print('CurrW - ' + str(currW))
    fcInputSize = int((currH*currW*inDepth))
    with tf.variable_scope('fc'+str(fcLayerNum-1)):
 #       print("input - " +str(fcInputSize) +" output - " + str(fcLayersDepths[fcLayerNum-1]))
        currInput = FCLayer(currInput,fcInputSize,fcLayersDepths[fcLayerNum-1],is_training,keep_prob=1.0,activation=None).output()
    y_out=currInput
    #print(shape(y_out))
    return y_out
