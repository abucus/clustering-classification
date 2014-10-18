'''
Created on Sep 29, 2014

@author: merlin-teng
'''
import unittest

import numpy as np
import os.path as op


class Test(unittest.TestCase):


    def testNPSaveArrayToFile(self):
        arr = np.array([[1, 3], [5, 6]])
#         np.savetxt("./a/test.txt", arr)
        isFile = op.isfile("output.ini")
        if(op.exists("./output.ini")):
            file = open("output.ini", "r+")
            reltCount = int(file.readline())
            file.seek(0, 0)
            file.write(str(reltCount + 1))
        else:
            file = open("output.ini", "w")
            file.write("2")
            reltCount = 1
        
        
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testNPSaveArrayToFile']
    unittest.main()
