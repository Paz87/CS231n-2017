# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:52:35 2017

@author: pazilan
"""
import tensorflow as tf

reg_str = 1e-8
def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

class ConvLayer(object):
    def __init__(self,input,filt_size,input_h,input_w,in_size,out_size,max_pool_on,batch_norm_on,is_training,keep_prob = 1.0,activation = 'relu', padding='SAME'):
        self.input = input      
        self.filt_size = filt_size
        self.input_h = input_h
        self.input_w = input_w
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.padding = padding
        self.batch_norm_on = batch_norm_on
        self.max_pool_on = max_pool_on
        self.is_training = is_training
        self.keep_prob = keep_prob
#        tf.cond(is_training,lambda:keep_prob,lambda:1.0)
#        self.W = tf.get_variable("W", shape=[filt_size, filt_size,in_size,out_size],
#           initializer=tf.contrib.layers.xavier_initializer_conv2d())
#        self.b = tf.get_variable(name="b", shape=[out_size], initializer=tf.zeros_initializer())
        
    def output(self):
#        new_size = [-1, self.input_h, self.input_w, self.in_size]
#        input_reshape = tf.reshape(self.input,new_size)
#        temp_out = tf.nn.conv2d(input_reshape, self.W, strides=[1,1,1,1], padding=self.padding) + self.b
#        
#        if self.activation == "relu":
#            temp_out = tf.nn.relu(temp_out)
#
#        if self.batch_norm_on:
#            temp_out = tf.contrib.layers.batch_norm(temp_out,center=True, scale=True,
#                                          is_training=self.is_training,
#                                          scope='bn')
#        if self.max_pool_on:
#            temp_out = tf.contrib.layers.max_pool2d(temp_out , [2,2], padding='SAME')
#        
#        temp_out = tf.nn.dropout(temp_out,self.keep_prob)
#            
#        self.output = temp_out
#        return self.output
        if self.activation == "relu":
            temp_out = tf.layers.conv2d(
                    inputs=self.input,
                    filters=self.out_size,
                    kernel_size=[self.filt_size,self.filt_size],
                    padding="SAME",
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    activity_regularizer = tf.contrib.layers.l2_regularizer(reg_str))
        else:
            temp_out= tf.layers.conv2d(
                    inputs=self.input,
                    filters=self.out_size,
                    kernel_size=[self.filt_size,self.filt_size],
                    padding="SAME",
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    activity_regularizer = tf.contrib.layers.l2_regularizer(reg_str))
        if self.batch_norm_on:
            temp_out = tf.contrib.layers.batch_norm(temp_out,center=True, scale=True,
                                          is_training=self.is_training,decay=0.9,
                                          scope='bn')
#            Second option to fix the bug - decrease decay
#             temp_out = tf.cond(
#                     self.is_training,
#                     lambda: tf.contrib.layers.batch_norm(temp_out, scope= 'bn', center=True, scale=True, is_training=True, reuse=None),
#                     lambda: tf.contrib.layers.batch_norm(temp_out, scope= 'bn', center=True, scale=True, is_training=False, reuse=True),
#                     )
        if self.max_pool_on:
            temp_out = tf.contrib.layers.max_pool2d(temp_out , [2,2], padding='SAME')
        
        temp_out = tf.layers.dropout(inputs=temp_out, rate=1-self.keep_prob, training=self.is_training) 
#                temp_out = tf.nn.dropout(temp_out,self.keep_prob)

            
        self.output = temp_out
        return self.output

class FCLayer(object):
    def __init__(self,input,input_size,output_size,is_training,keep_prob = 1.0,activation = 'relu'):
        self.input = input
        self.input_size = input_size
        self.output_size = output_size
#        self.keep_prob = tf.cond(is_training,lambda:keep_prob,lambda:1.0)
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.activation = activation
        self.W = tf.get_variable("W",shape=[input_size,output_size],initializer = tf.contrib.layers.xavier_initializer(),regularizer = tf.contrib.layers.l2_regularizer(reg_str))
        self.b = tf.get_variable("b",shape=[output_size],initializer = tf.zeros_initializer())
        
    def output(self):
      #  print(shape(self.input))
        new_size = [-1, self.input_size]
        input_reshape = tf.reshape(self.input,new_size)
        temp_out = tf.matmul(input_reshape,self.W)+self.b

        if self.activation == "relu":
            temp_out = tf.nn.relu(temp_out)
  
        temp_out = tf.layers.dropout(inputs=temp_out, rate=1-self.keep_prob, training=self.is_training)# == tf.estimator.ModeKeys.TRAIN)
#        temp_out = tf.nn.dropout(temp_out,self.keep_prob)
        
        self.output = temp_out
        #print(shape(self.output))
        return self.output
    
        