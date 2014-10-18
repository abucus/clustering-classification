'''
Created on Sep 14, 2014

@author: merlin-teng
'''
from sklearn import linear_model
from sklearn.svm import OneClassSVM
import unittest

import numpy as np


class Test(unittest.TestCase):


    def testLogisticRegression(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [2, 1], [3, 2], [4, 3], [5, 4]], dtype=np.float64)
        y = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        logreg = linear_model.LogisticRegression()
        logreg.fit(X, y)
        coef = logreg.coef_
        print coef, type(coef)
        print logreg.intercept_
        
        print 'result', logreg.decision_function(np.array([[3, 8]]))
        
        
        import matplotlib.pyplot as plt
        # plt.plot([0,1],[-1.04259691,1.04259691-1.04259691], marker='o', color='b', ls='')
        plt.plot(*zip(*(X)), marker='o', color='b', ls='')
        plt.plot([0, 20], [coef[0, 0], 20 * coef[0, 1] + coef[0, 0]])
        plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testLogisticRegression']
    unittest.main()
