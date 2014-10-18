'''
Created on Oct 5, 2014

@author: merlin-teng
'''
import unittest
import os.path as op
from classfier.clustering_selection_svm_classifier import ClusteringSelectionSVMClassifier
from data.demo_data import ellipse_data, readFromCSV, readDataFromFile
from validation.kfold_validate import KFoldValidation


class Test(unittest.TestCase):


    def testKFoldClusterSelectionSVM(self):
        data = ellipse_data()
        classifier = ClusteringSelectionSVMClassifier(clusterNum=3)
        
        v = KFoldValidation()
        relt = v.validate(data, classifier)
        print relt
        
    def testRealDataNS(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "nssample.csv"))
        c1 = ClusteringSelectionSVMClassifier(clusterNum=3, outputPath=None)
        
        v = KFoldValidation(basePath="../../ns_selection_svm_c3")
        relt = v.validate(data, c1, foldNum=10)
        print relt
    
    def testRealDataXD(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "xdsample.csv"))
        c1 = ClusteringSelectionSVMClassifier(clusterNum=3)
        
        v = KFoldValidation()
        relt = v.validate(data, c1, foldNum=10)
        print relt
    
    def testEllipseData3(self):
        data = readDataFromFile(op.join("..","..","28","init_data.txt"),op.join("..","..","28","init_labels.txt"))
        classifier = ClusteringSelectionSVMClassifier(clusterNum=2, outputPath=op.join("..", ".."), needScale=False)
        
        v = KFoldValidation(basePath=op.join("..","..","relt","svmselection_28_0"))
        relt = v.validate(data, classifier)

        print relt

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testKFoldClusterSelectionSVM']
    unittest.main()
