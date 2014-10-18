'''
Created on Oct 1, 2014

@author: merlin-teng
'''
import unittest

from classfier.clustering_logistic_classifier import ClusteringLogisticClassifier
from data.demo_data import demo, ellipse_data, readDataFromFile
import os.path as op


class Test(unittest.TestCase):
        
    def testClusteringClassifier(self):
        demoData = demo()
        data = demoData['data']
        labels = demoData['labels']

        clusteringClassifier = ClusteringLogisticClassifier(numOfClusters=2, outputPath=op.join("..", ".."))
        clusteringClassifier.fit(data, labels)
    
    def testClusteringClassifier2(self):
        demoData = readDataFromFile(op.join("..", "..", "50", "init_data.txt"), op.join("..", "..", "50", "init_labels.txt"))
        #demoData = antiKmeans()
        data = demoData['data']
        labels = demoData['labels']

        clusteringClassifier = ClusteringLogisticClassifier(numOfClusters=2, outputPath=op.join("..", ".."), needScale=False)
        clusteringClassifier.fit(data, labels)
        
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testClusteringClassifier']
    unittest.main()
