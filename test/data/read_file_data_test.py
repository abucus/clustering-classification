'''
Created on Oct 5, 2014

@author: merlin-teng
'''
import unittest

from data.demo_data import readDataFromFile
import os.path as op


class Test(unittest.TestCase):


    def testReadDataFromFile(self):
        data = readDataFromFile(op.join("..", "..", "129", "init_data.txt"), op.join("..", "..", "129", "init_labels.txt"))
        print data['data'].shape
        print data['labels'].shape


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testReadDataFromFile']
    unittest.main()
