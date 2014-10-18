'''
Created on Oct 4, 2014

@author: merlin-teng
'''
import unittest

from classfier.clustering_svm_classifier_with_init_cluster import ClusteringSVMClassifierWithInitCluster
from data.demo_data import ellipse_data, readFromCSV, readFromCSVWithoutScale,\
    readDataFromFile
import os.path as op
from validation.kfold_validate import KFoldValidation


class Test(unittest.TestCase):


    def testKFoldSVMInit(self):
        data = ellipse_data()
        classifier = ClusteringSVMClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        
        v = KFoldValidation(basePath="../../svminit")
        relt = v.validate(data, classifier)
        print relt

    def testRealDataNS(self):
        for i in range(5):
            data = readFromCSVWithoutScale(op.join("..", "..", "relt","validationdata", "nssample.csv"))
            c1 = ClusteringSVMClassifierWithInitCluster(numOfClusters=3, outputPath=op.join("..", ".."), numOfInitCluster=50)
            
            v = KFoldValidation(basePath=op.join("..","..","ns_svm_init_"+(str(i))))
            relt = v.validate(data, c1, foldNum=10)
            print relt
    
    def testRealDataXD(self):
        for i in range(5):
            data = readFromCSVWithoutScale(op.join("..", "..", "relt","validationdata", "xdsample.csv"))
            c1 = ClusteringSVMClassifierWithInitCluster(numOfClusters=3, outputPath=op.join("..", ".."), numOfInitCluster=50)
            
            v = KFoldValidation(basePath=op.join("..","..","xd_svm_init_"+(str(i))))
            relt = v.validate(data, c1, foldNum=10)
            print relt
            
    def testEllipseData3(self):
        data = readDataFromFile(op.join("..","..","28","init_data.txt"),op.join("..","..","28","init_labels.txt"))
        classifier = ClusteringSVMClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), needScale=False, numOfInitCluster=8)
        
        v = KFoldValidation(basePath=op.join("..","..","relt","svmInit_28_2"))
        relt = v.validate(data, classifier)

        print relt

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testKFoldSVMInit']
    unittest.main()
