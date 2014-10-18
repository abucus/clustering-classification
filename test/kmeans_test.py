'''
Created on Sep 26, 2014

@author: merlin-teng
'''
from numpy.random import random
from sklearn.cluster import KMeans
import unittest

import numpy as np


class Test(unittest.TestCase):


    def testKMeans(self):
        circleCenters = np.array([[1, 3], [2, 1], [5, 1], [6, 3]], np.float64)
        pointsNumPerCircle = 100
        pointsNum = 400
        data = np.empty([pointsNum, 2], dtype=np.float64)
        for i in range(0, 4):
            center = circleCenters[i]
            for j in range(pointsNumPerCircle * i , pointsNumPerCircle * (i + 1)):
                angle = random() * np.pi * 2
                radius = random()
                # print j,data[j,1],data[j][1],p[0]
                data[j, 0] = center[0] + radius * np.cos(angle)
                data[j, 1] = center[1] + radius * np.sin(angle)
        kmeans = KMeans(n_clusters=4)
        
        kmeans.fit(data)
        # print kmeans.cluster_centers_
        predict_relt = kmeans.predict([1, 3.001])
        print "1:", type(predict_relt), predict_relt
    
    def testKeans3D(self):
        circleCenters = np.array([[1, 3, 0], [2, 1, 0], [5, 1, 0], [6, 3, 0]], np.float64)
        pointsNumPerCircle = 100
        pointsNum = 400
        data = np.empty([pointsNum, 3], dtype=np.float64)
        for i in range(0, 4):
            center = circleCenters[i]
            for j in range(pointsNumPerCircle * i , pointsNumPerCircle * (i + 1)):
                angle = random() * np.pi * 2
                radius = random()
                # print j,data[j,1],data[j][1],p[0]
                data[j, 0] = center[0] + radius * np.cos(angle)
                data[j, 1] = center[1] + radius * np.sin(angle)
                data[j, 2] = center[2] + radius * np.sin(angle)
        kmeans = KMeans(n_clusters=4)
        
        kmeans.fit(data)
        # print kmeans.cluster_centers_
        
        predictRelt = kmeans.predict([[2, 3, 5], [4, 4, 6]]) 
        print type(predictRelt), predictRelt


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testKMeans']
    unittest.main()
