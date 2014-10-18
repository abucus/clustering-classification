'''
Created on Oct 7, 2014

@author: merlin-teng
'''
import unittest
import os.path as op
from data.demo_data import readFromCSV


class Test(unittest.TestCase):


    def testReadCSVData(self):
        demo = readFromCSV(op.join("/home", "merlin-teng", "nssample.csv"))
        data = demo['data']
        labels = demo['labels']
        print data.shape
        print labels.shape


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testReadCSVData']
    unittest.main()
