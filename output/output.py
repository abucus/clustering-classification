import matplotlib.pyplot as plt
import numpy as np
import os.path as path
from output_config import OutputConfig


class Output(object):
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    # markers = ["o", "x", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "D", "d", "|", "_"]
    markers = ["o", "x", "*", "^", "h", "s", "D", "+" , ">", "p" , "H", "d", "|", "_", "v"]
    axisrange = [-2, 10]
    def __init__(self, basePath, show=False, save=False, colorRange=None, markerRange=None):
        self.show = show
        self.save = save
        self.colorRange = colorRange
        self.markerRange = markerRange
        self.config = OutputConfig(basePath=basePath)
        self.baseOutputPath = self.config.prepareOutputDir()
    
    def outputInitFig(self, data, labels):
        if(self.show is not True and self.save is not True):
            return
        plt.clf()
        labelList = np.unique(labels)
        for i in range(len(labelList)):
            label = labelList[i]
            color = Output.colors[i]            
            plt.plot(*zip(*(data[labels == label])), marker="o", color=color, ls='')
        plt.xlim(Output.axisrange)
        plt.ylim(Output.axisrange)
        plt.savefig(path.join(self.baseOutputPath, "init_label_fig.png"))
        
    
    def outputFig(self, data, colorData, markerData, count=None, colorRange=None, markerRange=None, models=None, optionalTitle=""):
        plt.clf()
        if(self.show is not True and self.save is not True):
            return
        if(colorRange is None):
            colorRange = self.colorRange
        if(markerRange is None):
            markerRange = self.markerRange
        assert (colorRange is not None) and (markerRange is not None)
        assert np.shape(colorData)[0] == np.shape(markerData)[0]
        
        if(isinstance(colorRange, (int, long))):
            colorRange = range(colorRange)
        if(isinstance(markerRange, (int, long))):
            markerRange = range(markerRange)
        
        for i in colorRange:
            color = Output.colors[i]
            for j in markerRange:
                marker = Output.markers[j]
                idxes = ((colorData == i) & (markerData == j))
                plt.plot(*zip(*(data[idxes])), marker=marker, color=color, ls='')
                
            if(models is not None and models[i] is not None):
                model = models[i]
                a, b, c = a, b, c = model.coef_[0, 0], model.coef_[0, 1], model.intercept_[0]
                b += 1e-10 
                xx = np.arange(Output.axisrange[0], Output.axisrange[1], 0.01)
                yy = -a / b * xx - c / b
                plt.plot(xx, yy, color=color)
        plt.xlim(Output.axisrange)
        plt.ylim(Output.axisrange)
        if(self.save is True):
            if(count is not None):
                fileName = "iter_" + str(count) + "_fig_" + optionalTitle + ".png"
            else:
                fileName = "fig_" + optionalTitle + ".png"
            plt.savefig(path.join(self.baseOutputPath, fileName))
        if(self.show is True):
            plt.show(block=False)
            
    def saveData(self, data, fileName):
        np.savetxt(path.join(self.baseOutputPath, fileName), data)
        
    def saveInitClusterData(self, data):
        self.saveData(data, "init_clusters.txt")
        
    def saveInitData(self, data):
        self.saveData(data, "init_data.txt")
    
    def saveInitLabel(self, data):
        self.saveData(data, "init_labels.txt")
        
    def saveIterateCluster(self, data, count):
        self.saveData(data, "iter_" + str(count) + "_clusters.txt")
    
    def saveIterateCoefs(self, models, count):
        coefs = []
        for l in range(np.shape(models)[0]):
            m = models[l]
            coefs_for_m = []
            for i in m.coef_[0]:
                coefs_for_m.append(i)
            coefs_for_m.append(m.intercept_[0])
            coefs.append(coefs_for_m)
        self.saveData(coefs, "iter_" + str(count) + "_coefs.txt")
        
            
        
