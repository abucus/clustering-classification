'''
Created on Oct 5, 2014

@author: merlin-teng
'''
import unittest
import os.path as op
from classfier.clustering_selection_logistic_classifier import ClusteringSelectionLogisticClassifier
from data.demo_data import ellipse_data, readFromCSV, ellipse_data2,\
    readDataFromFile
from validation.kfold_validate import KFoldValidation


class Test(unittest.TestCase):


    def testKFoldClusteringSelectionLogistic(self):
        data = ellipse_data()
        classifier = ClusteringSelectionLogisticClassifier(clusterNum=3, outputPath=None)
        
        v = KFoldValidation(basePath=op.join("..","..","selection_logistic_relt"))
        relt = v.validate(data, classifier)
        print relt

    def testRealDataNS(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "nssample.csv"))
        c1 = ClusteringSelectionLogisticClassifier(clusterNum=5)
        
        v = KFoldValidation()
        relt = v.validate(data, c1, foldNum=10)
        print relt
    
    def testRealDataXD(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "xdsample.csv"))
        c1 = ClusteringSelectionLogisticClassifier(clusterNum=3)
        
        v = KFoldValidation()
        relt = v.validate(data, c1, foldNum=10)
        print relt
        
    def testEllipseData2(self):
        data = ellipse_data2()
        classifier = ClusteringSelectionLogisticClassifier(clusterNum=2, outputPath=op.join("..", ".."), needScale=False)
        
        v = KFoldValidation(basePath=op.join("..","..","relt","selectLogistic"))
        relt = v.validate(data, classifier)

        print relt
    
    def testEllipseData3(self):
        data = readDataFromFile(op.join("..","..","28","init_data.txt"),op.join("..","..","28","init_labels.txt"))
        classifier = ClusteringSelectionLogisticClassifier(clusterNum=2, outputPath=op.join("..", ".."), needScale=False)
        
        v = KFoldValidation(basePath=op.join("..","..","relt","logisticSelection_28_0"))
        relt = v.validate(data, classifier)

        print relt

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testKFoldClusteringSelectionLogistic']
    unittest.main()
