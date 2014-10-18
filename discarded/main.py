from numpy.random import randint
from sklearn import linear_model

from data.demo_data import Label
from data.demo_data import demo
import numpy as np
from output.output import Output


# Step 1 Initialize Data
demoData = demo()
data = demoData['data']
labels = demoData['labels']
numOfClusters = 2
clusters = randint(0, numOfClusters, data.shape[0])  # np.arange(0,400)<200
logregs = np.empty(numOfClusters, np.object)
lossValues = np.empty(numOfClusters, np.float64)
skipped = np.array([False] * numOfClusters)
maxIterateNum = 10

# init output
out = Output(basePath="..", colorRange=numOfClusters, markerRange=[Label.negative, Label.positive])
out.saveInitData(data)
out.saveInitClusterData(clusters)
# Step 2 Iterate to find the best classifier until some conditions are met
# for now is maxIterateNum
for i in range(maxIterateNum):
    print "Iteration Count:", i
    for j in range(numOfClusters):
        # Step 2.1 find all data belongs to current cluster
        print "Find Classifier for Clustering ", j
        indexes = np.where(clusters == j)
        clusterData = data[indexes]
        clusterLabels = labels[indexes]
        print "clusterData size:", clusterData.shape[0]
        # print "clusterLabels:\n",clusterLabels
        
        # if current cluster only include one label, logistic regression not available
        # just skip to next loop
        size0 = clusterLabels[clusterLabels == Label.negative].size
        print size0, " label0 in cluster ", j
        if clusterData.shape[0] == data.shape[0]:
            break
        if size0 == 0 or size0 == clusterLabels.size:
            skipped[j] = True
            print "skip"
            continue
        else:
            skipped[j] = False
        # Step 2.2 use logistic regression to generate a classifier for current cluster
        # LR(x,y) => w
        logreg = linear_model.LogisticRegression(C=1e10)
        logreg.fit(clusterData, clusterLabels)
        coef = logreg.coef_[0]
        print "for iteration ", i, " clustering ", j, " coef:", coef
        logregs[j] = logreg
        # print "for iteration ",i," clustering ",j," coef:",coef
    if np.all(skipped) or clusterData.shape[0] == data.shape[0]:
        print "break at ", i, "th iteration"
        break
       
    # Step 2.3 using loss function to assign points to its best classifier
    # to form new cluster
    for k in range(data.shape[0]):
        for l in range(numOfClusters):
            lossValues[l] = np.log(1 + np.exp(-logregs[l].predict(data[k]) * labels[k])) / np.log(2)
        # print np.min(lossValues)
        # print logregs[l].coef_
        # print data[k]," lostValues:",lossValues," decision vaue:",logregs[l].decision_function(data[k])
        clusters[k] = np.argmin(lossValues)
        
    # Draw result
    out.outputFig(data, clusters, labels, i, models=logregs)
    out.saveIterateCluster(clusters, i)
    out.saveIterateCoefs(logregs, i)

