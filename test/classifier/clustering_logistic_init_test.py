'''
Created on Oct 2, 2014

@author: merlin-teng
'''
import unittest

from classfier.clustering_logistic_classifier_with_init_cluster import ClusteringLogisticClassifierWithInitCluster
from data.demo_data import demo, ellipse_data, readDataFromFile, readFromCSV,\
    readFromCSVWithoutScale, ellipse_data2, antiKmeans
import os.path as op
import discarded.clustering_logistic_classifier_with_init_cluster as dc


class Test(unittest.TestCase):


    def testClusteringLogisticKmean(self):
        demoData = demo()
        data = demoData['data']
        labels = demoData['labels']

        classifier = ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        classifier.fit(data, labels)
        
        print classifier.predict([[1, 1], [4, 4], [7, 3]])
    
    def testClusteringLogisticKmean2(self):
        demoData = readDataFromFile("/home/merlin-teng/draw/init_data.txt", "/home/merlin-teng/draw/init_labels.txt")
        data = demoData['data']
        labels = demoData['labels']

        classifier = dc.ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        classifier.fit(data, labels)
        
        # print classifier.predict([[1, 1], [4, 4], [7, 3]])
        
    def testNoneVote(self):
        demoData = readDataFromFile(op.join("..", "..", "150", "init_data.txt"), op.join("..", "..", "145", "init_labels.txt"))
        data = demoData['data']
        labels = demoData['labels']
        
        c1 = ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        c1.fit(data, labels)
        
    def testVote(self):
        demoData = readDataFromFile(op.join("..", "..", "150", "init_data.txt"), op.join("..", "..", "145", "init_labels.txt"))
        data = demoData['data']
        labels = demoData['labels']
        c2 = dc.ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        c2.fit(data, labels)
        
    def testRealDataNS(self):
        demoData = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "nssample.csv"))
        data = demoData['data']
        labels = demoData['labels']
        c2 = ClusteringLogisticClassifierWithInitCluster(numOfClusters=5, outputPath=op.join("..", ".."), numOfInitCluster=60)
        c2.fit(data, labels)
        
    def testRealDataXD(self):
        demoData = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "xdsample.csv"))
        data = demoData['data']
        labels = demoData['labels']
        c2 = ClusteringLogisticClassifierWithInitCluster(numOfClusters=3, outputPath=op.join("..", ".."), numOfInitCluster=50)
        c2.fit(data, labels)
        
    def testInconsistancy(self):
        demoData = readDataFromFile(op.join("..", "..", "145", "init_data.txt"), op.join("..", "..", "145", "init_labels.txt"))
        data = demoData['data']
        labels = demoData['labels']

        c = dc.ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        c.fit(data, labels)
    
    def testEllipseData2(self):
        demoData = ellipse_data2()
        data = demoData['data']
        labels = demoData['labels']

        classifier = ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8, needScale=False)
        classifier.fit(data, labels)
        
    def testInconsistancy2(self):
        #demoData = readDataFromFile(op.join("..", "..", "50", "init_data.txt"), op.join("..", "..", "50", "init_labels.txt"))
        demoData = antiKmeans()
        data = demoData['data']
        labels = demoData['labels']

        c = dc.ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        c.fit(data, labels)
        
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testClusteringLogisticKmean']
    unittest.main()
