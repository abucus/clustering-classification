'''
Created on Oct 5, 2014

@author: merlin-teng
'''
import unittest

from classfier.clustering_svm_classifier import ClusteringSVMClassifier
from data.demo_data import ellipse_data, readFromCSV, readFromCSVWithoutScale,\
    readDataFromFile
import os.path as op
from validation.kfold_validate import KFoldValidation


class Test(unittest.TestCase):


    def testKFoldClusterSVM(self):
        data = ellipse_data()
        classifier = ClusteringSVMClassifier(numOfClusters=2, outputPath=op.join("..", ".."))
        
        v = KFoldValidation()
        relt = v.validate(data, classifier)
        print relt

    def testRealDataNS(self):
        data = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "nssample.csv"))
        #for i in range(5):
        data = readFromCSV(op.join("..", "..", "validationdata", "nssample.csv"))
        c1 = ClusteringSVMClassifier(numOfClusters=5, outputPath=op.join("..", ".."))
        
        v = KFoldValidation(op.join("..","..","ns_svm3"))
        relt = v.validate(data, c1, foldNum=10)

    
    def testRealDataXD(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "xdsample.csv"))
        c1 = ClusteringSVMClassifier(numOfClusters=3, outputPath=op.join("..", ".."))
        
        v = KFoldValidation(op.join("..","..","xd_svm2"))
        relt = v.validate(data, c1, foldNum=10)
        print relt
    
    def testEllipseData3(self):
        data = readDataFromFile(op.join("..","..","28","init_data.txt"),op.join("..","..","28","init_labels.txt"))
        classifier = ClusteringSVMClassifier(numOfClusters=2, outputPath=op.join("..", ".."), needScale=False)
        
        v = KFoldValidation(basePath=op.join("..","..","relt","svm_28_2"))
        relt = v.validate(data, classifier)

        print relt
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testKFoldClusterSVM']
    unittest.main()
