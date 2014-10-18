'''
Created on Oct 4, 2014

@author: merlin-teng
'''
from sklearn.linear_model import LogisticRegression
import unittest
import os.path as op
from data.demo_data import ellipse_data, readFromCSV, readFromCSVWithoutScale,\
    readDataFromFile
from validation.kfold_validate import KFoldValidation


class Test(unittest.TestCase):


    def testSimpleLogistic(self):
        
        data = ellipse_data()
        classifier = LogisticRegression(C=1e10)
        
        v = KFoldValidation(basePath="../../simple_relt",output=False)
        relt = v.validate(data, classifier)
        print relt
        
    def testRealDataNS(self):
        data = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "nssample.csv"))
        c1 = LogisticRegression(C=1e10)
        
        v = KFoldValidation(output=False)
        relt = v.validate(data, c1, foldNum=10)
        print relt
    
    def testRealDataXD(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "xdsample.csv"))
        c1 = LogisticRegression(C=1e10)
        
        v = KFoldValidation(output=False)
        relt = v.validate(data, c1, foldNum=10)
        print relt

    def testEllipseData3(self):
        data = readDataFromFile(op.join("..","..","28","init_data.txt"),op.join("..","..","28","init_labels.txt"))
        classifier = LogisticRegression()
        
        v = KFoldValidation(basePath=op.join("..","..","relt","logistic_simple_28_2"), output=False)
        relt = v.validate(data, classifier)

        print relt

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSimpleLogistic']
    unittest.main()
