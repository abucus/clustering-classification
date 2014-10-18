'''
Created on Oct 5, 2014

@author: merlin-teng
'''
import unittest

from classfier.clustering_logistic_classifier import ClusteringLogisticClassifier
from data.demo_data import ellipse_data, readFromCSV, readDataFromFile
import os.path as op
from validation.kfold_validate import KFoldValidation


class Test(unittest.TestCase):


    def testKFoldClusterLogistic(self):
        data = ellipse_data()
        classifier = ClusteringLogisticClassifier(numOfClusters=2, outputPath=op.join("..", ".."))
        
        v = KFoldValidation(basePath=op.join("..","..","kfold_logistic_relt"))
        relt = v.validate(data, classifier)
        print relt
        
    def testRealDataNS(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "nssample.csv"))
        c1 = ClusteringLogisticClassifier(numOfClusters=5, outputPath=op.join("..", ".."))
        
        v = KFoldValidation()
        relt = v.validate(data, c1, foldNum=10)
        print relt
    
    def testRealDataXD(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "xdsample.csv"))
        c1 = ClusteringLogisticClassifier(numOfClusters=3, outputPath=op.join("..", ".."))
        
        v = KFoldValidation()
        relt = v.validate(data, c1, foldNum=10)
        print relt
        
    def testEllipseData3(self):
        data = readDataFromFile(op.join("..","..","28","init_data.txt"),op.join("..","..","28","init_labels.txt"))
        classifier = ClusteringLogisticClassifier(numOfClusters=3, outputPath=op.join("..", ".."), needScale=False)
        
        v = KFoldValidation(basePath=op.join("..","..","relt","logistic_28_2"))
        relt = v.validate(data, classifier)

        print relt

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testKFoldClusterLogistic']
    unittest.main()
