from numpy.random import randint
from sklearn import linear_model
from sklearn.cluster import KMeans

from data.demo_data import Label
from data.demo_data import demo
import matplotlib.pyplot as plt
import numpy as np


# Step 1 Initialize Data
demoData = demo()
data = demoData['data']

# generate a init grid here
initClustersNum = 4
initClusters = KMeans(n_clusters=initClustersNum).fit_predict(data)


labels = demoData['labels']
numOfClusters = 2
clusters = randint(0, numOfClusters, data.shape[0])  # np.arange(0,400)<200
logregs = np.empty(numOfClusters, np.object)
lossValues = np.empty(numOfClusters, np.float64)
skipped = np.array([False] * numOfClusters)
maxIterateNum = 10
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]
print clusters
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
#     plt.plot(*zip(*(data[clusters == 0])), marker='o', color='b', ls='')
#     plt.plot(*zip(*(data[clusters == 1])), marker='o', color='r', ls='')
#     plt.show(block=False)
#     time.sleep(1)
#     plt.close()

    # Step 2.4 adjust clusters according to init clusterNum
    votes = np.zeros(numOfClusters)
    for m in range(initClustersNum):
        indexes = np.where(initClusters == m)[0]
        currenClusters = clusters[indexes]
        for n in range(numOfClusters):
            votes[n] = np.where(currenClusters == n)[0].size
        voteRelt = np.argmax(votes)
        clusters[indexes] = voteRelt
    # Draw
#     for m in range(numOfClusters):
#         negativeIdx = ((clusters) == m & (labels == Label.negative))
#         positiveIdx = ((clusters) == m & (labels == Label.positive))
#         plt.plot(*zip(*(data[negativeIdx])), marker='o', color=colors[m], ls='')
#         plt.plot(*zip(*(data[positiveIdx])), marker='+', color=colors[m], ls='')
#         a,b,c = logregs[m].coef_[0,0],logregs[m].coef_[0,1],logregs[m].intercept_[0]
#         b += 1e-10 
#         xx = np.arange(0,7,0.01)
#         yy = -a/b*xx-c/b
#         plt.plot(xx,yy,color=colors[m])
#         print m, (data[negativeIdx]).shape, (data[positiveIdx]).shape
# #     print clusterData
#     plt.show()
#     for m in range(numOfClusters):
#         color = str(m*0.25)
#         plt.plot(*zip(*(data[clusters == m])), marker='o', color=colors[m], ls='')
#         a,b,c = logregs[m].coef_[0,0],logregs[m].coef_[0,1],logregs[m].intercept_[0]
#         b += 1e-10 
#         xx = np.arange(0,7,0.01)
#         yy = -a/b*xx-c/b
#         plt.plot(xx,yy,color=colors[m])
#         plt.xlim([0,8])
#         plt.ylim([0,8])
#     plt.show()
    
    for m in range(numOfClusters):
        color = colors[m]
        for n in range(initClustersNum):
            marker = markers[n]
            indexes = ((clusters == m) & (initClusters == n))
            plt.plot(*zip(*(data[indexes])), marker=marker, color=color, ls='')
        a, b, c = logregs[m].coef_[0, 0], logregs[m].coef_[0, 1], logregs[m].intercept_[0]
        b += 1e-10 
        xx = np.arange(0, 7, 0.01)
        yy = -a / b * xx - c / b
        plt.plot(xx, yy, color=colors[m])
        plt.xlim([0, 8])
        plt.ylim([0, 8])
    plt.show()
    
print clusters
for m in range(numOfClusters):
    color = str(m * 0.25)
    plt.plot(*zip(*(data[clusters == m])), marker='o', color=colors[m], ls='')
    a, b, c = logregs[m].coef_[0, 0], logregs[m].coef_[0, 1], logregs[m].intercept_[0]
    b += 1e-10 
    xx = np.arange(0, 7, 0.01)
    yy = -a / b * xx - c / b
    plt.plot(xx, yy, color=colors[m])
plt.show()


