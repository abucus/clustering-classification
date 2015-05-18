'''
Created on Oct 1, 2014

@author: merlin-teng
'''
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC

from classfier.base_clustering_classifier import BaseClusteringClassifier


class ClusteringSVMClassifier(BaseClusteringClassifier):
    '''
    classdocs
    '''


    def __init__(self, numOfClusters, outputPath, needScale=True):
        '''
        Constructor
        '''
        BaseClusteringClassifier.__init__(self, numOfClusters, outputPath, needScale)
        
    def _generate_classifier(self, x, y, kernel = 'linear'):
        sizeOfOneLabel = (y[y == self.labelList[0]].size)
        if sizeOfOneLabel == 0 or sizeOfOneLabel == y.size:
            classifier = OneClassSVM(kernel)
        else:
            classifier = SVC(C=1e10, kernel = kernel)
        classifier.fit(x, y)
        return classifier
    
    def _cal_loss_value(self, classifier, x, y):
        return max(1 - classifier.decision_function(x) * y, 0)
        
