'''
Created on Oct 8, 2014

@author: merlin-teng
'''
import unittest
from validation.kfold_validate import KFoldValidation


class Test(unittest.TestCase):


    def testOutputLabels(self):
        kf = KFoldValidation()
        kf._output_validation_labels([1, 2, 3, 4, 5], [6, 7, 8.0, 9.0, 10])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testOutputLabels']
    unittest.main()
