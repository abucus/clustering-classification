'''
Created on Sep 21, 2014

@author: merlin-teng
'''
from sklearn.svm import OneClassSVM
import unittest

import matplotlib.pyplot as plt
import numpy as np


class Test(unittest.TestCase):


    def testSVM(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [2, 1], [3, 2], [4, 3], [5, 4]], dtype=np.float64)
        Y = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        svm = OneClassSVM(kernel="linear")
        svm.fit(X, Y)
        print svm.coef_
        
        a, b, c = svm.coef_[0, 0], svm.coef_[0, 1], svm.intercept_
        XX = np.arange(0, 6, 0.01)
        YY = (-a * XX - c) / b
        
        plt.plot(X, Y)
        plt.plot(XX, YY)
        plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSVM']
    unittest.main()
