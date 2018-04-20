# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:49:11 2018

@author: Gergo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:50:56 2017

@author: Gergo
"""
import numpy as np
import pandas as pd
from collections import deque
import random
from copy import deepcopy

def softmax(a):
    x = np.asarray(a,dtype=float)
    exps = np.exp(x)
    return  exps/exps.sum()

class ExperienceMemory:

    def __init__(self, buffer_size, enable_per=True):
        self.buffer = pd.DataFrame([[{},\
                                     np.array([]),\
                                     np.array([]),\
                                     {},\
                                     False,\
                                     np.array([]),\
                                     0,\
                                     0.]],\
                                   columns=['s', 'a', 'r', "s'",'over','g','e','p'])
        self.buffer = self.buffer.drop(self.buffer.index[0])
        self.size = buffer_size
        self.enable_per = enable_per
        self.num_items = 0
        self.e = 0.01
        self.alfa = 0.7

    def size(self):
        return self.size

    def _getPriority(self, error):
        return (error + self.e) ** self.alfa

    def update_priorities(self,indeces,errors):
        priorities = self._getPriority(np.abs(errors))
        self.buffer.loc[indeces,'p'] = priorities

    def sample_experience(self,number_of_samples):
        if self.num_items < number_of_samples:
            batch = self.buffer
        else:
            if self.enable_per:
                p = self.buffer.loc[:,'p']
                p = p / np.sum(p)
                batch = self.buffer.sample(number_of_samples,replace=False,weights = p)
            else:
                batch = self.buffer.sample(number_of_samples,replace=False)
        return batch.values[:],batch.index.values
    def get_last_episode(self,length):
        batch = self.buffer.iloc[self.num_items-length:]
        return batch.values[:]
    def add_experience(self,item):
#        item.loc[:,'p']=self._getPriority(abs(item.loc[:,'r']))
        if self.size > self.num_items+item.shape[0]:
            self.buffer=pd.concat([self.buffer,item],ignore_index=True)
        else:
            self.buffer = self.buffer.drop(self.buffer.index[0:item.shape[0]])
            self.buffer = pd.concat([self.buffer,item],ignore_index=True)
        self.num_items = self.buffer.shape[0]


    def reset(self):
         self.buffer = self.buffer.drop(self.buffer.index[:])
         self.num_items = 0

