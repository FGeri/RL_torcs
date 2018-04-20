# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:14:30 2017

@author: Gergo
"""
#%%
import numpy as np
import math
import keras
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Add, Concatenate,BatchNormalization,Dropout
from keras.layers import Flatten, Input, merge, Lambda, Activation
from keras.layers import Conv2D,MaxPool2D,LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam,Nadam,RMSprop
import keras.backend as K
import tensorflow as tf



class Critic:
    def __init__(self, sess, state_size,action_size,LR):
        self.sess = sess
        self.LR = LR
        self.action_size=action_size
        K.set_learning_phase(1)
        K.set_session(sess)

        #Now create the model
        self.model, self.state_1,self.state_2, self.action = self.create_network(state_size,action_size)
        self.target_model, self.target_state_1,self.target_state_2, self.target_action = self.create_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state_1: states['state_2D'],
            self.state_2: states['state_1D'],
            self.action: actions
        })[0]

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())


    def create_network(self, state_size,action_size):

#        512x256
        inp_1 = Input(shape=state_size['state_2D'],name='state_1')
        inp_2 = Input(shape=state_size['state_1D'],name='state_2')
        inp_3 = Input(shape=action_size,name='action')

        branch_1 = Conv2D(filters=32,kernel_size=3,strides=2,activation="relu")(inp_1)
        branch_1 = Conv2D(filters=32,kernel_size=3,strides=1,activation="relu")(branch_1)
        branch_1 = Conv2D(filters=32,kernel_size=3,strides=1,activation="relu")(branch_1)
        branch_1 = Flatten()(branch_1)

        branch_12 = Concatenate()([branch_1,inp_2,inp_3])
        branch_12 = Dense(64, activation='relu')(branch_12)
        branch_12 = Dense(64, activation='relu')(branch_12)
        branch_12 = Dense(32, activation='relu')(branch_12)
        V = Dense(1,activation='linear')(branch_12)
        model = Model(inputs=[inp_1,inp_2,inp_3],outputs=V)
        optimizer = Adam(lr=self.LR)
        model.compile(loss='mse', optimizer=optimizer)
        return model, inp_1,inp_2,inp_3