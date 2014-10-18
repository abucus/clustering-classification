'''
Created on Oct 1, 2014

@author: merlin-teng
'''
import unittest

from classfier.clustering_svm_classifier import ClusteringSVMClassifier
from data.demo_data import demo, ellipse_data
import os.path as op


class Test(unittest.TestCase):


    def testClusteringSVMClassifier(self):
        demoData = demo()
        data = demoData['data']
        labels = demoData['labels']

        classifier = ClusteringSVMClassifier(numOfClusters=2, outputPath=op.join("..", ".."))
        classifier.fit(data, labels)
    
    def testClusteringSVMClassifier2(self):
        demoData = ellipse_data()
        data = demoData['data']
        labels = demoData['labels']

        classifier = ClusteringSVMClassifier(numOfClusters=2, outputPath=op.join("..", ".."))
        classifier.fit(data, labels)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testClusteringSVMClassifier']
    unittest.main()
