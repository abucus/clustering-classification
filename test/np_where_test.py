'''
Created on Sep 14, 2014

@author: merlin-teng
'''
import unittest

import numpy as np


class Test(unittest.TestCase):


    def testNPWhere(self):
        print type(np)
        x = np.array([[0, 1], [1, 2], [2, 3]])
        y = np.array([True, True, False, True])
        relt = np.where(y == False)[0]
        print type(relt), relt.size
        print x[np.where(y == False)[0]]

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testNPWhere']
    unittest.main()
