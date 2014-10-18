'''
Created on Sep 14, 2014

@author: merlin-teng
'''
import unittest

from data.demo_data import demo
import matplotlib.pyplot as plt


class Test(unittest.TestCase):


    def testDemoData(self):
        demoData = demo()
        data = demoData['data']
        plt.plot(*zip(*(data)), marker='o', color='b', ls='')
        plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
