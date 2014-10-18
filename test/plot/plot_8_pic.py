'''
Created on Oct 8, 2014

@author: merlin-teng
'''
import unittest
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
from sklearn.cluster import KMeans
from data.demo_data import Label
from output.output import Output
from matplotlib.pyplot import cm


class Test(unittest.TestCase):


    def testSimplePlot(self):
        clusterNum = 2
        basePath = op.join("..","..","50")
        data = np.loadtxt(op.join(basePath, "init_data.txt"), dtype=np.float64)
        labels = np.loadtxt(op.join(basePath, "init_labels.txt"), dtype=np.int8)
        positiveLabels = (labels == Label.positive)
        negativeLabels = (labels == Label.negative)
        segments = np.loadtxt(op.join(basePath, "init_segment.txt"), dtype=np.int8)
        
        
        rowNum = clusterNum
        colNum = 3
        markers = ["o", "x", "*", "^", "h", "s", "D", "+" , ">", "p" , "H", "d", "|", "_", "v"]
        fileNums = [0, 1, 9]
        plotIdx = 1
        #plt.figure(figsize=(8, 6),dpi=80)
        #plt.suptitle("Process of Classification", fontsize=14, fontweight='bold')
        for i in range(rowNum):
            for j in range(colNum):
                clusters = np.loadtxt(op.join(basePath,"iter_" + str(fileNums[j]) + "_clusters.txt"))
                coeffs = np.loadtxt(op.join(basePath,"iter_" + str(fileNums[j]) + "_coefs.txt"))
                clusterIdxes = clusters == i
                
                sp = plt.subplot(rowNum, colNum, plotIdx)
                sp.set_xlim(Output.axisrange)
                sp.set_ylim(Output.axisrange)
                sp.set_xlabel("$x_1$")
                sp.set_ylabel("$x_2$")
                print "i,j:",i,j,"cluster len:",len(clusters),op.join(basePath,"iter_" + str(fileNums[j]) + "_clusters.txt")
                # draw point
                for k in range(8):
                    segmentIdxes = (segments == k)
                    positiveDataCurSegment = data[(clusterIdxes & positiveLabels & segmentIdxes)]
                    print i,j,k,len(positiveDataCurSegment)
                    if(len(positiveDataCurSegment) > 0):
                        sp.scatter(*zip(*(positiveDataCurSegment)), marker=markers[k], facecolor="r", edgecolor="r")
                        
                    negativeDataCurSegment = data[(clusterIdxes & negativeLabels & segmentIdxes)]
                    if(len(negativeDataCurSegment) > 0):
                        sp.scatter(*zip(*(negativeDataCurSegment)), marker=markers[k], facecolor="b", edgecolor="b")
                # draw line   
                a, b, c = coeffs[i, 0], coeffs[i, 1], coeffs[i, 2]
                b += 1e-10 
                xx = np.arange(Output.axisrange[0], Output.axisrange[1], 0.01)
                yy = -a / b * xx - c / b
                sp.plot(xx, yy, color="k")
                        
                plotIdx = plotIdx + 1 
        plt.subplot(rowNum, colNum, 1).set_title("Initialization")
        plt.subplot(rowNum, colNum, 2).set_title("Intermediate")
        plt.subplot(rowNum, colNum, 3).set_title("Final")
        #plt.subplot(rowNum, colNum, 1).set_ylabel("Cluster 1",fontsize=14)
        #plt.subplot(rowNum, colNum, 4).set_ylabel("Cluster 2",fontsize=14)
        plt.text(-.2, .5, 'Cluster 1',horizontalalignment='right',verticalalignment='center',rotation='vertical',transform=plt.subplot(rowNum, colNum, 1).transAxes, fontsize=14)
        plt.text(-.2, .5, 'Cluster 2',horizontalalignment='right',verticalalignment='center',rotation='vertical',transform=plt.subplot(rowNum, colNum, 4).transAxes, fontsize=14)
        #plt.tight_layout()
        plt.savefig("/home/merlin-teng/figure_1.pdf")
        plt.show()
    
    def testPlot2(self):
        basePath = op.join("..","..","50")
        data = np.loadtxt(op.join(basePath, "init_data.txt"), dtype=np.float64)
        labels = np.loadtxt(op.join(basePath, "init_labels.txt"), dtype=np.int8)
        plt.figure(figsize=(8,6))
        
        positiveData = data[labels == Label.positive]
        negativeData = data[labels == Label.negative]
        plt.xlim(Output.axisrange)
        plt.ylim(Output.axisrange)
        plt.scatter(*zip(*(positiveData)), marker="o", facecolor="r", edgecolor="r", label = "Positive")
        plt.scatter(*zip(*(negativeData)), marker="o", facecolor="b", edgecolor="b", label = "Negative")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()
        plt.show()
        pass
    
    def testPlot3(self):
        basePath = op.join("..","..","50")
        totalLoss = np.loadtxt(op.join(basePath, "TotalLoss.txt"), dtype = np.float64)
        plt.figure(figsize=(8,6))
        plt.ylim(45,85)
        plt.plot(range(10), totalLoss, "ks-", linewidth=2)
        plt.ylabel("Total Loss")
        plt.xlabel("Iteration")
        plt.show()
        
    def testHistogram(self):
        segmentIdx = range(1,5)
        segments = ["Segment "+str(a) for a in segmentIdx]
        data = np.array([432,946,1466,20], dtype=np.int32)
        plt.hist([432,946,1466,20])
        plt.xlim(0,3000)
        plt.ylim(0,3000)
        #plt.xticks(segmentIdx, segments)
        plt.show()

    def testInconsistancy(self):
        basePath = op.join("..","..","81")
        data = np.loadtxt(op.join(basePath, "init_data.txt"), dtype=np.float64)
        clusterBeforeVote = np.loadtxt(op.join(basePath, "iter_4_clusters_beforeVote"))
        
        plt.figure(figsize=(8, 6),dpi=80)
        plt.xlim(Output.axisrange)
        plt.ylim(Output.axisrange)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.scatter(*zip(*(data[clusterBeforeVote == 0])), marker="^", facecolor="b", edgecolor="b", label="Cluster 1")
        plt.scatter(*zip(*(data[clusterBeforeVote == 1])), marker="o", facecolor="r", edgecolor="r", label="Cluster 2")
        plt.legend()
        plt.show()
        
    def testInconsistancy2(self):
        basePath = op.join("..","..","81")
        data = np.loadtxt(op.join(basePath, "init_data.txt"), dtype=np.float64)
        clusterAfterVote = np.loadtxt(op.join(basePath, "iter_4_clusters.txt"))
        plt.figure(figsize=(8,6),dpi=80)
        plt.xlim(Output.axisrange)
        plt.ylim(Output.axisrange)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.scatter(*zip(*(data[clusterAfterVote == 0])), marker="^", facecolor="b", edgecolor="b", label="Cluster 1")
        plt.scatter(*zip(*(data[clusterAfterVote == 1])), marker="o", facecolor="r", edgecolor="r", label="Cluster 2")
        plt.legend()
        plt.show()
        
    def plotKmeans(self):
        basePath = op.join("..","..","drawdata","ellipse_2_8pic")
        data = np.loadtxt(op.join(basePath, "init_data.txt"), dtype=np.float64)
        kmean = KMeans(n_clusters=2)
        relt = kmean.fit_predict(data)
        plt.scatter(*zip(*(data[relt == 0])), marker="^", facecolor="b", edgecolor="b", label="Cluster 1")
        plt.scatter(*zip(*(data[relt == 1])), marker="o", facecolor="r", edgecolor="r", label="Cluster 2")
        plt.show()
        
    def plotColorMapping(self):
        data = np.random.rand(4,4)
        fig, ax = plt.subplots()
        data = np.loadtxt(op.join("..","..","drawdata","heatmap","NS_iter_9_coefs.txt"),dtype=np.float64)
        data = abs(data[:,0:49])
        heatmap = ax.pcolor(data, cmap=plt.get_cmap("Blues"))
        row_labels = [ "Segment "+str(x) for x in range(1,6)]
        column_labels = [""]*49
        # put the major ticks at the middle of each cell
        #ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
        
        # want a more natural, table-like display
        #ax.invert_yaxis()
        #ax.xaxis.tick_top()
        
        ax.set_xticklabels(column_labels, minor=False)
        ax.set_yticklabels(row_labels, minor=False, fontsize=18)
        ax.set_xlabel('49 Profile Attributes', fontsize=18)
        plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSimplePlot']
    unittest.main()
