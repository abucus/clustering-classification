'''
Created on Sep 14, 2014

@author: merlin-teng
'''
import unittest

import numpy as np


class Test(unittest.TestCase):


    def testNPSimple(self):
        labels = np.empty(10, dtype=np.int8)
        print labels


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testNPSimple']
    unittest.main()
