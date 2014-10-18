'''
Created on Oct 10, 2014

@author: merlin-teng
'''
import unittest
from util.myIO import DictionaryIO
import os.path as op
class Test(unittest.TestCase):


    def testDictionaryIOWrite(self):
        mi = DictionaryIO()
        mi.attr("list", [2,3,4])
        mi.attr("float", 1.23)
        mi.attr("str","string")
        mi.writeToFile(op.join("..","..","testFile.pkl"))
        pass
    
    def testDictionaryIORead(self):
        mi = DictionaryIO()
        mi.loadFromFile(op.join("..","..","testFile.pkl"))
        print mi.attr("list"),type(mi.attr("list"))
        print mi.attr("float"),type(mi.attr("float"))
        print mi.attr("str"),type(mi.attr("str"))
        
    def testDictionaryIOWrite2(self):
        mi = DictionaryIO()
        mi.attr("int", 9)
        mi.writeToFile(op.join("..","..","testFile.pkl"))
        pass
    
    def testDictionaryIORead2(self):
        mi = DictionaryIO()
        mi.loadFromFile(op.join("..","..","testFile.pkl"))
        for key,value in mi.dic.items():
            print key,"=",value
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDictionaryIO']
    unittest.main()