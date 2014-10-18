'''
Created on Oct 5, 2014

@author: merlin-teng
'''
from sklearn import linear_model

from classfier.base_clustering_selection_classifier import BaseClusteringSelectionClassifier


class ClusteringSelectionLogisticClassifier(BaseClusteringSelectionClassifier):
    '''
    classdocs
    '''


    def __init__(self, clusterNum, outputPath, needScale=True):
        '''
        Constructor
        '''
        BaseClusteringSelectionClassifier.__init__(self, clusterNum, outputPath, needScale)
    
    def _generate_classifier(self, x, y):
        classifier = linear_model.LogisticRegression(C=1e10)
        classifier.fit(x, y)
        return classifier
