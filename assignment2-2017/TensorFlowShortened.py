# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 01:17:18 2017

@author: pazilan
"""


import sys 
sys.path.append("C:\\Users\\pazilan\\Desktop\\CS231n\\assignment2-2017")
import os
os.chdir("C:\\Users\\pazilan\\Desktop\\CS231n\\assignment2-2017")
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import TensorFlowLayers
from TensorFlowLayers import ConvLayer,FCLayer
import importlib
from importlib import reload

# In[ ]:


from cs231n.data_utils import load_CIFAR10
chkpt_file = '../CIFAR10_data/cifar_cnn.ckpt'
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    print(training_now)
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
            
        if training_now:
            save_path = saver.save(sess, chkpt_file)
            print("Model saved in file: %s" % save_path)
    return total_loss,total_correct


# In[ ]:
reload(TensorFlowLayers)
from TensorFlowLayers import *
import SimpleNet
reload(SimpleNet)
from SimpleNet import my_model
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X,y,is_training)
total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss) + tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
optimizer = tf.train.AdamOptimizer(1e-4) # select optimizer and set learning rate 1e-4

pass

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

saver = tf.train.Saver()

# In[ ]:
restore_call = False;
sess = tf.Session()

if restore_call:
            # Restore variables from disk.
            saver.restore(sess, chkpt_file) 
else: 
    sess.run(tf.global_variables_initializer())
    
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,2,128,10,train_step,True)
print('Training result')
run_model(sess,y_out,mean_loss,X_train,y_train,1,64)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
print('Test')
run_model(sess,y_out,mean_loss,X_test,y_test,1,64)


sess.close()


# In[]:
restore_call = True;
sess = tf.Session()
if restore_call:
            # Restore variables from disk.
            saver.restore(sess, chkpt_file) 
else: 
    sess.run(tf.global_variables_initializer())
    
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,20,128,10,train_step,True)
print('Training result')
run_model(sess,y_out,mean_loss,X_train,y_train,1,64)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
print('Test')
run_model(sess,y_out,mean_loss,X_test,y_test,1,64)
sess.close()


