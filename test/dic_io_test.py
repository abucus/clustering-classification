'''
Created on Sep 29, 2014

@author: merlin-teng
'''
import unittest

from util.myIO import DictionaryIO 


class Test(unittest.TestCase):


    def testDicWrite(self):
        io = DictionaryIO()
        io.attr("id", 5)
        io.attr("name", "Mingfei")
        io.writeToFile("a.txt")
        
    def testDictRead(self):
        io = DictionaryIO()
        io.loadFromFile("a.txt")
        print io.attr("id"), io.attr("name")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testDicIO']
    unittest.main()
