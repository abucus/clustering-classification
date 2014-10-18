'''
Created on Oct 4, 2014

@author: merlin-teng
'''
from sklearn.svm import LinearSVC
import unittest
import os.path as op
from data.demo_data import ellipse_data, readFromCSV, readDataFromFile
from validation.kfold_validate import KFoldValidation


class Test(unittest.TestCase):


    def testSimpleSVM(self):
        data = ellipse_data()
        classifier = LinearSVC(C=1e10, loss="l1")
        
        v = KFoldValidation()
        relt = v.validate(data, classifier)
        print relt
    
    def testRealDataNS(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "nssample.csv"))
        c1 = LinearSVC(C=1e10, loss="l1")
        
        v = KFoldValidation(output=False)
        relt = v.validate(data, c1, foldNum=10)
        print relt
    
    def testRealDataXD(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "xdsample.csv"))
        c1 = LinearSVC(C=1e10, loss="l1")
        
        v = KFoldValidation(output=False)
        relt = v.validate(data, c1, foldNum=10)
        print relt
    
    def testEllipseData3(self):
        data = readDataFromFile(op.join("..","..","28","init_data.txt"),op.join("..","..","28","init_labels.txt"))
        classifier = LinearSVC(C=1e10, loss="l1")
        
        v = KFoldValidation(basePath=op.join("..","..","relt","svmselection_28_0"),output=False)
        relt = v.validate(data, classifier)

        print relt
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSimpleSVM']
    unittest.main()
