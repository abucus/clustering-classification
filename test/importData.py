# from numpy import recfromcsv, genfromtxt
# from sklearn.feature_extraction import DictVectorizer
# from data.demo_data import readDataFromFile
# import matplotlib.pyplot as plt
# import numpy as np
import csv
import sys

filePath = "/home/merlin-teng/Dropbox/SDM2015/B2Bdata/reduced/NS.csv"
 
# demo = recfromcsv(filePath)
#  
# for i in demo.dtype.names:
#     col = demo[i]
#     for j in range(len(col)):
#         if(col[j] is None):
#             print i, " ", j
#      
# vec = DictVectorizer()
# relt = vec.fit_transform(demo).toarray()
# print relt


# Draw 
# data = np.loadtxt('../145/init_data.txt', dtype=np.float64)
# labels = np.loadtxt('../145/init_labels.txt', dtype=np.int8)
# colors = ['#00FF00','#9900CC']
# plt.clf()
# labelList = np.unique(labels)
# for i in range(len(labelList)):
#     label = labelList[i]
#     color = colors[i]          
#     plt.plot(*zip(*(data[labels == label])), marker="o", color=color, ls='')
# plt.xlim([0, 9])
# plt.ylim([0, 9])
# plt.show()
reader = csv.reader(open(filePath, 'r'), delimiter=',')
try:
    for row in reader:
        # print row,type(row)
        print row
        break
except csv.Error as e:
    sys.exit('file %s, line %d: %s' % (filePath, reader.line_num, e))

