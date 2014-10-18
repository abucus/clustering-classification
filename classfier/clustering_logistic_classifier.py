'''
Created on Oct 1, 2014

@author: merlin-teng
'''
from sklearn import linear_model

from classfier.base_clustering_classifier import BaseClusteringClassifier
import numpy as np


class ClusteringLogisticClassifier(BaseClusteringClassifier):
    '''
    classdocs
    '''


    def __init__(self, numOfClusters, outputPath, needScale=True):
        '''
        Constructor
        '''
        BaseClusteringClassifier.__init__(self, numOfClusters, outputPath, needScale)
        
    def _generate_classifier(self, x, y):
        classifier = linear_model.LogisticRegression(C=1e10)
        classifier.fit(x, y)
        return classifier
    
    def _cal_loss_value(self, classifier, x, y):
        return np.log(1 + np.exp(-classifier.decision_function(x) * y)) / np.log(2)
    
    def _is_skip_needed(self, j, skipped, clusterData, clusterLabels):
        sizeOf1stLabel = clusterLabels[clusterLabels == self.labelList[0]].size
        if sizeOf1stLabel == 0 or sizeOf1stLabel == clusterLabels.size:
            skipped[j] = True
        else:
            skipped[j] = False
        return skipped[j]
    
    def _is_stop_needed(self, data, clusterData):
        return clusterData.shape[0] == data.shape[0]
