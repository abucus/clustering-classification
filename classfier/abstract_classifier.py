'''
Created on Oct 1, 2014

@author: merlin-teng
 
'''
from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractClassifier(object):
    '''
    This class is an abstract class which define the basic behavior of a classifier
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def fit(self, x, y):
        '''
        Using the input x and y to train a model
        '''
        pass
    
    @abstractmethod
    def predict(self, x):
        pass
    
    def _checkData(self, x):
        assert len(np.shape(x)) > 0
        if(len(np.shape(x)) == 1):
            x = np.expand_dims(x, axis=0)
        else:
            x = np.array(x)
        return x
            
        
