'''
Created on Oct 2, 2014

@author: merlin-teng
'''
import unittest

from classfier.clustering_svm_classifier_with_init_cluster import ClusteringSVMClassifierWithInitCluster
from data.demo_data import demo, ellipse_data, readDataFromFile, readFromCSV,\
    readFromCSVWithoutScale
import os.path as op
import discarded.clustering_svm_classifier_with_init_cluster as dc

class Test(unittest.TestCase):


    def testClusteringSVMKmeanInit(self):
        demoData = demo()
        data = demoData['data']
        labels = demoData['labels']

        classifier = ClusteringSVMClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        classifier.fit(data, labels)
    
    def testClusteringSVMKmeanInit2(self):
        demoData = ellipse_data()
        data = demoData['data']
        labels = demoData['labels']

        classifier = ClusteringSVMClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        classifier.fit(data, labels)

    def testNoneVote(self):
        demoData = ellipse_data()
        data = demoData['data']
        labels = demoData['labels']
        
        c1 = ClusteringSVMClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        c1.fit(data, labels)
        
    def testVote(self):
        demoData = readDataFromFile(op.join("..", "..", "181", "init_data.txt"), op.join("..", "..", "145", "init_labels.txt"))
        data = demoData['data']
        labels = demoData['labels']
        c2 = dc.ClusteringSVMClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        c2.fit(data, labels)
        
    def testRealDataNS(self):
        demoData = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "nssample.txt"))
        data = demoData['data']
        labels = demoData['labels']
        c2 = ClusteringSVMClassifierWithInitCluster(numOfClusters=5, outputPath=op.join("..", ".."), numOfInitCluster=60)
        c2.fit(data, labels)
        
    def testRealDataXD(self):
        demoData = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "xdsample.txt"))
        data = demoData['data']
        labels = demoData['labels']
        c2 = ClusteringSVMClassifierWithInitCluster(numOfClusters=3, outputPath=op.join("..", ".."), numOfInitCluster=50)
        c2.fit(data, labels)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testClusteringSVMKmeanInit']
    unittest.main()
