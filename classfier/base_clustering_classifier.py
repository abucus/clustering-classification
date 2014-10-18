'''
Created on Oct 1, 2014

@author: merlin-teng
'''
from abc import abstractmethod
from numpy import argmin
from numpy.random import randint
import os.path as op
from classfier.abstract_classifier import AbstractClassifier
import numpy as np
from output.output import Output
from util.distance import distance_point_to_hyperplane
from sklearn.preprocessing import StandardScaler

class BaseClusteringClassifier(AbstractClassifier):
    '''
    classdocs
    '''

    def __init__(self, numOfClusters, outputPath = op.join("..",".."), needScale=True):
        AbstractClassifier.__init__(self)
        self.numOfClusters = numOfClusters
        self.outputPath = outputPath
        self.needScale = needScale
    
    def _is_skip_needed(self, j, skipped, clusterData, clusterLabels):
        return False
    
    def _is_stop_needed(self, data, clusterData):
        return False
    
    @abstractmethod
    def _generate_classifier(self, x, y):
        pass
    
    @abstractmethod
    def _cal_loss_value(self, classifier, x, y):
        pass
    
    def fit(self, x, y):        
        # Step 1 Initialize Data
        if(self.needScale):
            scaler = self.scaler = StandardScaler()
            data = scaler.fit_transform(x)
        else:
            data = x
        labels = y
        maxIterateNum = 10
        numOfClusters = self.numOfClusters
        clusters = randint(0, numOfClusters, data.shape[0])  # np.arange(0,400)<200
        lossValues = np.empty(numOfClusters, np.float64)
        classifiers = self._classifiers = np.empty(numOfClusters, np.object)
        skipped = np.array([False] * numOfClusters)
        labelList = self.labelList = np.unique(y)
        
        # init output
        out = Output(basePath=self.outputPath, colorRange=self.numOfClusters, markerRange=labelList)
        out.saveInitData(data)
        out.saveInitClusterData(clusters)
        out.saveInitLabel(labels)
        
        # Step 2 Iterate to find the best classifier until some conditions are met
        # for now is maxIterateNum
        for i in range(maxIterateNum):
            for j in range(numOfClusters):
                # Step 2.1 find all data belongs to current cluster
                indexes = np.where(clusters == j)
                clusterData = data[indexes]
                clusterLabels = labels[indexes]
                print "for iteration ", i, " clustering ", j, "clusterData size:", clusterData.shape[0]
                
                if self._is_stop_needed(data, clusterData):
                    print "break at iteration ", i, " cluster ", j
                    break
                if self._is_skip_needed(j, skipped, clusterData, clusterLabels):
                    continue
                    
                # Step 2.2 generate a classifier for current cluster
                # LR(x,y) => w
                classifier = self._generate_classifier(clusterData, clusterLabels)
                classifiers[j] = classifier
                # print "for iteration ",i," clustering ",j," coef:",coef
            else:
                
                # Step 2.3 using loss function to assign points to its best classifier
                # to form new cluster
                for k in range(data.shape[0]):
                    for l in range(numOfClusters):
                        lossValues[l] = self._cal_loss_value(classifiers[l], data[k], labels[k])
                    # print data[k]," lostValues:",lossValues," decision vaue:",classifiers[l].decision_function(data[k])
                    clusters[k] = np.argmin(lossValues)
                    
                # Draw result
                out.outputFig(data, clusters, labels, i, models=classifiers)
                out.saveIterateCluster(clusters, i)
                out.saveIterateCoefs(classifiers, i)
                continue
            break
    
    def predict(self, x):
        x = self._checkData(x)
        if(self.needScale):
            x = self.scaler.transform(x)
        classifiers = self._classifiers
        labels = np.zeros(len(x), dtype=np.int8)
        distances = np.zeros(len(classifiers))
        
        for i in range(len(x)):
            p0 = x[i]
            distances.fill(0)
            for j in range(len(classifiers)):
                c = classifiers[j]
                coefs = c.coef_[0]
                intercept = c.intercept_[0]
                dist = distance_point_to_hyperplane(p0, coefs, intercept) 
                if(dist == 0):
                    labels[i] = c.predict(x[i])
                    break
                else:
                    distances[j] = dist
            else:
                labels[i] = classifiers[argmin(distances)].predict(x[i])
        return labels
