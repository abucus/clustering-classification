'''
Created on Sep 29, 2014

@author: merlin-teng
'''
import pickle
import os.path as op
class DictionaryIO(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.dic = {}
        
    def attr(self, attrName, attrValue=None):
        if(attrValue is not None):
            self.dic[attrName] = attrValue
        else:
            return self.dic[attrName]
    
    def writeToFile(self, path):
        with open(path, 'w') as output:
            pickle.dump(self.dic, output, pickle.HIGHEST_PROTOCOL)
    
    def loadFromFile(self, path):
        if(op.exists(path) is False):
            return self
        with open(path, 'r') as input:
            self.dic = pickle.load(input)
        return self
