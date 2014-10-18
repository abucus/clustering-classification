'''
Created on Sep 29, 2014

@author: merlin-teng
'''
import os

import os.path as op 
from util.myIO import DictionaryIO
from global_all.const_variable import Constants

class OutputConfig(object):
    '''
    classdocs
    '''

    def __init__(self, basePath):
        '''
        Constructor
        '''
        self.basePath = basePath
        
        dicIO = DictionaryIO()
        path = op.join(basePath, Constants.OUT_PUT_CONFIG_FILE_NAME)
        if(op.exists(path)):
            dicIO.loadFromFile(path)
        else:
            dicIO.attr(Constants.CURRENT_RELT_NUMBER, 1)
        
        self.currentReltNO = dicIO.attr(Constants.CURRENT_RELT_NUMBER) 
        dicIO.attr(Constants.CURRENT_RELT_NUMBER, self.currentReltNO + 1)
        dicIO.writeToFile(path)
    
    def prepareOutputDir(self):
        outputDir = op.join(self.basePath, str(self.currentReltNO))
        if(op.exists(outputDir) is False):
            os.mkdir(outputDir)
        return outputDir
