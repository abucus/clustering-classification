'''
Created on Oct 2, 2014

@author: merlin-teng
'''
from numpy.random import randint
from sklearn.cluster import KMeans

from classfier.clustering_svm_classifier import ClusteringSVMClassifier
import numpy as np
from output.output import Output
from sklearn.preprocessing import StandardScaler

class ClusteringSVMClassifierWithInitCluster(ClusteringSVMClassifier):
    '''
    classdocs
    '''


    def __init__(self, numOfClusters, outputPath, numOfInitCluster, needScale = True):
        '''
        labelList is a argument not in use here
        '''
        ClusteringSVMClassifier.__init__(self, numOfClusters, outputPath, needScale)
        self.numOfInitCluster = numOfInitCluster
        
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
        lossValues = np.empty(numOfClusters, np.float64)
        accumulatedLoss = np.zeros(maxIterateNum)
        classifiers = self._classifiers = np.empty(numOfClusters, np.object)
        skipped = np.array([False] * numOfClusters)
        self.labelList = np.unique(y)
        
        initClustersNum = self.numOfInitCluster
        self._initKmeans = KMeans(n_clusters=initClustersNum)
        initClusters = self._initClusters = self._initKmeans.fit_predict(data)
        
        clusters = self._clusters = np.zeros(data.shape[0])  # np.arange(0,400)<200
        for i in range(initClustersNum):
            clusters[initClusters == i] = randint(0, numOfClusters)
            
        # init output
        # previous labelList is replaced by range(initClustersNum) here
        out = Output(basePath=self.outputPath, colorRange=self.numOfClusters, markerRange=range(initClustersNum))
        out.saveInitData(data)
        out.saveInitClusterData(clusters)
        out.saveInitLabel(labels)
        out.saveData(initClusters, "init_segment.txt")
        out.outputInitFig(data, labels)
        out.outputFig(data, clusters, initClusters, models=classifiers, optionalTitle="start")
        
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
                    print "skip for i ", i, " j", j
                    continue
                    
                # Step 2.2 generate a classifier for current cluster
                # LR(x,y) => w
                classifier = self._generate_classifier(clusterData, clusterLabels)
                # coef = classifier.coef_[0]
                # print "for iteration ", i, " clustering ", j, " coef:", coef
                classifiers[j] = classifier
            else:
                
                # Step 2.3 using loss function to assign points to its best classifier
                # to form new cluster
                out.outputFig(data, clusters, initClusters, i, models=classifiers, optionalTitle="step1_learn_model")
                for k in range(initClustersNum):
                    lossValues.fill(0)
                    idxes = np.where(initClusters == k)[0]
                    for idx in idxes:
                        for l in range(numOfClusters):
                            lossValues[l] = lossValues[l] + self._cal_loss_value(classifiers[l], data[idx], labels[idx])
                    clusters[idxes] = np.argmin(lossValues)
                    accumulatedLoss[i] += lossValues.min()
                
                # Draw result
                # olor data is clustering, markerData is kmean clusters
                out.outputFig(data, clusters, initClusters, i, models=classifiers, optionalTitle="step2_reassign_cluster")
                out.saveIterateCluster(clusters, i)
                out.saveIterateCoefs(classifiers, i)
                continue
            break
        out.saveData(accumulatedLoss, "TotalLoss.txt")
        
    def predict(self, x):
        x = self._checkData(x)
        if(self.needScale):
            x = self.scaler.transform()
        kClusters = self._initKmeans.predict(x)
        initClusters = self._initClusters
        relt = np.empty(len(x), dtype=np.int32)
        for i in range(len(x)):
            relt[i] = self._classifiers[self._clusters[initClusters == kClusters[i]][0]].predict(x[i])
        return relt
