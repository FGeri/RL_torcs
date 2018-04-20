# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:46:11 2017

@author: Gergo
"""
#%%
from keras.models import Sequential,Model,model_from_json
from keras.initializers import normal
from keras.layers import Dense, Input, Concatenate,Conv2D,MaxPool2D,Flatten
from keras.layers import Lambda,Activation,BatchNormalization,Dropout,LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K


class Actor:
    def __init__(self, sess, state_size,action_size,LR):
        self.sess = sess
        self.LR = LR

        K.set_learning_phase(1)
        K.set_session(sess)
        self.model , self.weights, self.state_1,self.state_2 = self.create_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state_1,self.target_state_2 = self.create_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, *action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)

        self.optimize = tf.train.AdamOptimizer(LR).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())


    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state_1: states['state_2D'],
            self.state_2: states['state_1D'],
            self.action_gradient: action_grads
        })

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def create_network(self,state_size,action_dim):
#        512x256
        inp_1 = Input(shape=state_size['state_2D'],name='state_1')
        inp_2 = Input(shape=state_size['state_1D'],name='state_2')

        branch_1 = Conv2D(filters=32,kernel_size=3,strides=2,activation="relu")(inp_1)
        branch_1 = Conv2D(filters=32,kernel_size=3,strides=1,activation="relu")(branch_1)
        branch_1 = Conv2D(filters=32,kernel_size=3,strides=1,activation="relu")(branch_1)
        branch_1 = Flatten()(branch_1)

        branch_12 = Concatenate()([branch_1,inp_2])
        branch_12 = Dense(64, activation='relu')(branch_12)
        branch_12 = Dense(64, activation='relu')(branch_12)
        branch_12 = Dense(32, activation='relu')(branch_12)
        Steering = Dense(1,activation='linear')(branch_12)
        Throttle = Dense(1,activation='linear')(branch_12)
        Brake = Dense(1,activation='sigmoid')(branch_12)
        Hand_brake = Dense(1,activation='sigmoid')(branch_12)
        Reverse = Dense(1,activation='sigmoid')(branch_12)
        V = Concatenate()([Steering,Throttle,Brake,Hand_brake,Reverse])
        model = Model(inputs=[inp_1,inp_2],outputs=V)
        return model, model.trainable_weights, inp_1, inp_2

