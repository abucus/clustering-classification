'''
Created on Oct 5, 2014

@author: merlin-teng
'''
from sklearn.svm.classes import OneClassSVM, LinearSVC

from classfier.base_clustering_selection_classifier import BaseClusteringSelectionClassifier
import numpy as np


class ClusteringSelectionSVMClassifier(BaseClusteringSelectionClassifier):
    '''
    classdocs
    '''


    def __init__(self, clusterNum, outputPath, needScale = True):
        '''
        Constructor
        '''
        BaseClusteringSelectionClassifier.__init__(self, clusterNum, outputPath, needScale)
    
    def _generate_classifier(self, x, y):
        labelList = np.unique(y)
        sizeOfOneLabel = (y[y == labelList[0]].size)
        if sizeOfOneLabel == 0 or sizeOfOneLabel == y.size:
            classifier = OneClassSVM(kernel="linear")
        else:
            classifier = LinearSVC(C=1e10, loss="l1")
        classifier.fit(x, y)
        return classifier
        
