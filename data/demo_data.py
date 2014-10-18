from numpy.random import random
from sklearn import preprocessing
import numpy as np
import csv
import sys
import itertools
from numpy import float64
class Label:
    negative = -1
    positive = 1
    
def demo():
    circleCenters = np.array([[1, 3], [2, 1], [5, 1], [6, 3]], np.float64)
    pointsNumPerCircle = 100
    pointsNum = 400
    data = np.empty([pointsNum, 2], dtype=np.float64)
    labels = np.empty(pointsNum, dtype=np.int8)
    for i in range(0, 4):
        center = circleCenters[i]
        for j in range(pointsNumPerCircle * i , pointsNumPerCircle * (i + 1)):
            angle = random() * np.pi * 2
            radius = random()
            # print j,data[j,1],data[j][1],p[0]
            data[j, 0] = center[0] + radius * np.cos(angle)
            data[j, 1] = center[1] + radius * np.sin(angle)

            if(i % 2 == 0):
                labels[j] = Label.positive
            else:
                labels[j] = Label.negative
    demo = {'data':data, 'labels':labels}
    return demo

def ellipse_data(size=400, weight=[.65, .35]):
    data = np.zeros([size, 2], dtype=np.float64)
    group1Size = int(size * weight[0])
    group2Size = size - group1Size
    data[0:group1Size] = np.random.multivariate_normal([2, 6], [[1, 0], [0, 0.5]], group1Size)
    data[group1Size:size] = np.random.multivariate_normal([6, 2], [[2, 0], [0, 1]], group2Size)
    labels = np.zeros(size, dtype=np.int8)
    for i in range(size):
        p = data[i]
        if(i < group1Size):
            if(0.5 * p[0] + p[1] >= 7):
                labels[i] = Label.positive
            else:
                labels[i] = Label.negative
        else:
            if(2 * p[0] + p[1] >= 14):
                labels[i] = Label.negative
            else:
                labels[i] = Label.positive
    demo = {'data':data, 'labels':labels}
    return demo

def ellipse_data2(size=280, weight=[.5, .5]):
    data = np.zeros([size, 2], dtype=np.float64)
    group1Size = int(size * weight[0])
    group2Size = size - group1Size
    data[0:group1Size] = np.random.multivariate_normal([2.1, 6], [[2, 0], [0, 1]], group1Size)
    data[group1Size:size] = np.random.multivariate_normal([5, 3], [[2, 0], [0, 1]], group2Size)
    labels = np.zeros(size, dtype=np.int8)
    for i in range(size):
        p = data[i]
        if(i < group1Size):
            if(0.5 * p[0] + p[1] >= 7.05):
                labels[i] = Label.positive
            else:
                labels[i] = Label.negative
        else:
            if(2 * p[0] + p[1] >= 13):
                labels[i] = Label.negative
            else:
                labels[i] = Label.positive
    demo = {'data':data, 'labels':labels}
    return demo

def ellipse_data3():
    size=280
    weight=[.5, .5]
    group1Size = int(size * weight[0])
    group2Size = size - group1Size
    dataList1 = np.random.multivariate_normal([2.1, 6], [[2, 0], [0, 1]], group1Size).tolist()
    dataList2 = np.random.multivariate_normal([5, 3], [[2, 0], [0, 1]], group2Size).tolist()
    dataList2 = [x for x in dataList2 if -0.5*x[0]+x[1]>0.5]
    data = np.array(dataList1+dataList2, dtype = np.float64)
    
    labels = np.zeros(size, dtype=np.int8)
    for i in range(data.shape[0]):
        p = data[i]
        if(i < group1Size):
            if(0.5 * p[0] + p[1] >= 7.05):
                labels[i] = Label.positive
            else:
                labels[i] = Label.negative
        else:
            if(2 * p[0] + p[1] >= 13):
                labels[i] = Label.negative
            else:
                labels[i] = Label.positive
    demo = {'data':data, 'labels':labels}
    return demo

def antiKmeans():
    center = [4.,4.]
    pointsNum = 600
    data = np.empty([pointsNum, 2], dtype=np.float64)
    labels = np.empty(pointsNum, dtype=np.int8)
    slope = np.array([np.tan(2.*np.pi/3.), np.tan(np.pi/3.)],dtype=float64)
    for i in range(pointsNum):
        angle = random() * np.pi * 2
        radius = 2.+random()*3
        
        x = data[i, 0] = center[0] + radius * np.cos(angle)
        y = data[i, 1] = center[1] + radius * np.sin(angle)
        
        if(-slope[0]*x+y <= (center[1]-center[0]*slope[0])):
            labels[i] = Label.positive
        else:
            if(-slope[1]*x+y <= (center[1]-center[0]*slope[1]) and (y>=4)):
                labels[i] = Label.positive
            else:
                labels[i] = Label.negative
               
        
        
    demo = {'data':data, 'labels':labels}
    return demo

def readDataFromFile(dataFilePath, labelFilePath):
    data = np.loadtxt(dataFilePath, dtype=np.float64)
    labels = np.loadtxt(labelFilePath, dtype=np.int8)
    demo = {'data':data, 'labels':labels}
    return demo

def readFromCSV(filePath, ignoreFirstRow=True, dtype=None):
    if(ignoreFirstRow):
        skip = True
    else:
        skip = False
        
    with open(filePath, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        records = []
        try:
            for row in reader:
                if(skip):
                    skip = False
                    continue
                records.append(map(float, row))
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filePath, reader.line_num, e))
    dataMatrix = np.array(records, dtype=np.float64)
    npdata = np.split(dataMatrix, [dataMatrix.shape[1] - 1], axis=1)
    return {'data':preprocessing.scale(npdata[0]), 'labels':npdata[1].flatten()}

def readFromCSVWithoutScale(filePath, ignoreFirstRow=True, dtype=None):
    if(ignoreFirstRow):
        skip = True
    else:
        skip = False
        
    with open(filePath, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        records = []
        try:
            for row in reader:
                if(skip):
                    skip = False
                    continue
                records.append(map(float, row))
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filePath, reader.line_num, e))
    dataMatrix = np.array(records, dtype=np.float64)
    npdata = np.split(dataMatrix, [dataMatrix.shape[1] - 1], axis=1)
    return {'data':npdata[0], 'labels':npdata[1].flatten()}


