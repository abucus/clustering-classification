'''
Created on Oct 5, 2014

@author: merlin-teng
'''
import unittest

from data.demo_data import ellipse_data, Label, ellipse_data2, ellipse_data3,\
    antiKmeans, readDataFromFile
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
from sklearn.cluster.k_means_ import KMeans
class Test(unittest.TestCase):


    def testEllipseData(self):
        demoData = ellipse_data()
        data = demoData['data']
        labels = demoData['labels']
        plt.plot(*zip(*(data[labels == Label.negative])), marker='o', color='b', ls='')
        plt.plot(*zip(*(data[labels == Label.positive])), marker='o', color='r', ls='')
        plt.xlim([-2, 10])
        plt.ylim([-2, 10])
        plt.show()
    
    def testEllipseData2(self):
        demoData = ellipse_data2()
        data = demoData['data']
        labels = demoData['labels']
        np.savetxt(op.join("..","..","drawdata","ellipse_data2","data.txt"), data)
        np.savetxt(op.join("..","..","drawdata","ellipse_data2","labels.txt"), data)
        plt.plot(*zip(*(data[labels == Label.negative])), marker='o', color='b', ls='')
        plt.plot(*zip(*(data[labels == Label.positive])), marker='o', color='r', ls='')
        plt.xlim([-2, 10])
        plt.ylim([-2, 10])
        plt.show()
        
    def testEllipseData3(self):
        demoData = antiKmeans()
        data = demoData['data']
        labels = demoData['labels']
        #np.savetxt(op.join("..","..","drawdata","ellipse_data2","data.txt"), data)
        #np.savetxt(op.join("..","..","drawdata","ellipse_data2","labels.txt"), data)
        plt.subplot(1,2,1)
        plt.plot(*zip(*(data[labels == Label.negative])), marker='o', color='b', ls='')
        plt.plot(*zip(*(data[labels == Label.positive])), marker='o', color='r', ls='')
        plt.xlim([-2, 10])
        plt.ylim([-2, 10])
        
        plt.subplot(1,2,2)
        kmean = KMeans(n_clusters=2)
        clusters = kmean.fit_predict(data)
        plt.plot(*zip(*(data[clusters==0])), marker='o', color='b', ls='')
        plt.plot(*zip(*(data[clusters==1])), marker='o', color='r', ls='')
        plt.xlim([-2, 10])
        plt.ylim([-2, 10])
        
        plt.show()
        
    def test50EllipseData3(self):
        data = readDataFromFile(op.join("..","..","50","init_data.txt"), op.join("..","..","50","init_labels.txt"))
        segment = np.loadtxt(op.join("..","..","50","init_segment.txt.txt"), dtype=np.int8)
        clustersBeforeVote = np.loadtxt(op.join("..","..","50","iter_1_clusters_beforeVote"), dtype=np.int8)
        clustersBeforeVote = np.loadtxt(op.join("..","..","50","iter_1_clusters_beforeVote"), dtype=np.int8)
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testEllipseData']
    unittest.main()
