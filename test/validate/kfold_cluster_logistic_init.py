'''
Created on Oct 4, 2014

@author: merlin-teng
'''
import unittest

from classfier.clustering_logistic_classifier_with_init_cluster import ClusteringLogisticClassifierWithInitCluster
from data.demo_data import ellipse_data, readFromCSV, readFromCSVWithoutScale,\
    ellipse_data2, readDataFromFile
from validation.kfold_validate import KFoldValidation
import os.path as op
import numpy as np
from util.myIO import DictionaryIO
from time import strftime
from global_all.const_variable import Constants
class Test(unittest.TestCase):


    def testKFoldLogisticInit(self):
        data = ellipse_data()
        classifier = ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8)
        
        v = KFoldValidation(basePath=op.join("..","..","testKFoldLogisticInit2"))
        relt = v.validate(data, classifier)

        print relt
    
    def testEllipseData2(self):
        data = readDataFromFile(op.join("..","..","28","init_data.txt"),op.join("..","..","28","init_labels.txt"))
        classifier = ClusteringLogisticClassifierWithInitCluster(numOfClusters=2, outputPath=op.join("..", ".."), numOfInitCluster=8, needScale=False)
        
        v = KFoldValidation(basePath=op.join("..","..","relt","logisticInit_28_0"))
        relt = v.validate(data, classifier)

        print relt
    
    def testFitModelToWholeSample(self):
        data = readFromCSV(op.join("..", "..", "validationdata", "nssample.csv"))
        c1 = ClusteringLogisticClassifierWithInitCluster(numOfClusters=5, outputPath=op.join("..", ".."), numOfInitCluster=60)
        
        v = KFoldValidation()
        relt = v.validate(data, c1, foldNum=10)
        
        dicIO = DictionaryIO()
        dicIO.loadFromFile(op.join("..", "..",Constants.OUT_PUT_CONFIG_FILE_NAME))
        alldata = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "NS.csv"))        
        bestModel = relt["models"][relt["best_model_idx"]]
        allSegments = bestModel._initKmeans.predict(bestModel.scaler.transform(alldata["data"]))
        
        best_idx_str = str(relt["best_model_idx"])
        timestap = strftime("%H:%M:%S")
        max_folder_str = str(dicIO.attr(Constants.CURRENT_RELT_NUMBER)-1)
        np.savetxt(op.join("..","..","logistic_init_ns_segments_maxfold_"+max_folder_str+"_bestidx_"+best_idx_str+"_"+timestap), allSegments)
        np.savetxt(op.join("..","..","logistic_init_ns_f1_maxfold_"+max_folder_str+"_bestidx_"+best_idx_str+"_"+timestap), relt["f1"])
        np.savetxt(op.join("..","..","logistic_init_ns_precision_maxfold_"+max_folder_str+"_bestidx_"+best_idx_str+"_"+timestap), relt["precision"])
        np.savetxt(op.join("..","..","logistic_init_ns_accuracy_maxfold_"+max_folder_str+"_bestidx_"+best_idx_str+"_"+timestap), relt["accuracy"])
        np.savetxt(op.join("..","..","logistic_init_ns_recall_maxfold_"+max_folder_str+"_bestidx_"+best_idx_str+"_"+timestap), relt["recall"])
        np.savetxt(op.join("..","..","logistic_init_ns_average_precision_maxfold_"+max_folder_str+"_bestidx_"+best_idx_str+"_"+timestap), relt["average_precision"])
        np.savetxt(op.join("..","..","logistic_init_ns_roc_auc_maxfold_"+max_folder_str+"_bestidx_"+best_idx_str+"_"+timestap), relt["roc_auc"])
        
        
    def testRealDataXD(self):
        
        data = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "xdsample.csv"))
        for i in range(5):
            c1 = ClusteringLogisticClassifierWithInitCluster(numOfClusters=5, outputPath=op.join("..", ".."), numOfInitCluster=60)
            
            v = KFoldValidation(op.join("..","..","ns_logistic_init_"+str(i)))
            relt = v.validate(data, c1, foldNum=10)
            print relt     
    
    def testRealDataNS(self):
        data = readFromCSVWithoutScale(op.join("..", "..", "validationdata", "nssample.csv"))
        for i in range(5):
            c1 = ClusteringLogisticClassifierWithInitCluster(numOfClusters=3, outputPath=op.join("..", ".."), numOfInitCluster=50)
            
            v = KFoldValidation(op.join("..","..","ns_logistic_init_"+str(i)))
            relt = v.validate(data, c1, foldNum=10)
            print relt

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testKFold']
    unittest.main()
