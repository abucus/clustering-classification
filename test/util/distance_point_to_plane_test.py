'''
Created on Oct 5, 2014

@author: merlin-teng
'''
import unittest

import numpy as np
from util.distance import distance_point_to_hyperplane


class Test(unittest.TestCase):

    def testDistancePointToPlane(self):
        print distance_point_to_hyperplane(np.array([0, 0]), np.array([1, 1]), -1)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testDistancePointToPlane']
    unittest.main()
