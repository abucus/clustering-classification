'''
Created on Oct 5, 2014

@author: merlin-teng
'''
from abc import ABCMeta, abstractmethod
from sklearn.cluster.k_means_ import KMeans

from classfier.abstract_classifier import AbstractClassifier
import numpy as np
import os.path as op
from output.output import Output
from sklearn.preprocessing import StandardScaler
class BaseClusteringSelectionClassifier(AbstractClassifier):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta

    def __init__(self, clusterNum=5, outputPath = op.join("..",".."), needScale=True):
        '''
        Constructor
        '''
        self.clusterNum = clusterNum
        self.outputPath = outputPath
        self.needScale = needScale
    
    @abstractmethod
    def _generate_classifier(self, x, y):
        pass
    
    def fit(self, x, y):
        '''
        Using the input x and y to train a model
        '''
        out = Output(basePath=self.outputPath, show=False, save=False)
        if(self.needScale):
            scaler = self.scaler = StandardScaler()
            x = scaler.fit_transform(x)
        clusterNum = self.clusterNum
        kmean = self._kmean = KMeans(n_clusters=clusterNum)
        clusters = kmean.fit_predict(x)
        models = self._models = np.empty(clusterNum, dtype=np.object)
        for i in range(clusterNum):
            idxes = (clusters == i)
            data = x[idxes]
            labels = y[idxes]
            models[i] = self._generate_classifier(data, labels)
        out.saveInitClusterData(clusters)
        out.saveIterateCoefs(models, 0)
        pass
    
    def predict(self, x):
        if(self.needScale):
            x = self.scaler.transform(x)
        clusters = self._kmean.predict(x)
        labels = np.zeros(len(clusters), dtype=np.int8)
        for i in range(len(clusters)):
            labels[i] = self._models[clusters[i]].predict(x[i])
        return labels
