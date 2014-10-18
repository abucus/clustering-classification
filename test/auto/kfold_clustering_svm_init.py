from data.demo_data import readFromCSV
import os.path as op
import logging
from classfier.clustering_svm_classifier_with_init_cluster import ClusteringSVMClassifierWithInitCluster
from validation.kfold_validate import KFoldValidation
from util.myIO import DictionaryIO
from global_all.const_variable import Constants
configPath = op.join("..", "..", Constants.OUT_PUT_CONFIG_FILE_NAME)
csvFiles = ["nssample.csv"]  # , #"xdsample.csv"]
workingLog = op.join("..", "..", "autojob.log")
outputPath = op.join("/home", "merlin-teng", "kfold_validation_relt.txt")
# logging.basicConfig(filename=workingLog,level=logging.DEBUG)
clusterNumRange = range(2, 6)
numOfInitClusterRange = range(10, 70, 10)
iterNumRange = range(1)
dicIO = DictionaryIO()

for csvFile in csvFiles:
    data = readFromCSV(op.join("/home", "merlin-teng", csvFile))
    for i in iterNumRange:
        for j in clusterNumRange:
            for k in numOfInitClusterRange:
                try:
                    outputFolderNum = dicIO.loadFromFile(configPath).attr(Constants.CURRENT_RELT_NUMBER)
                    c1 = ClusteringSVMClassifierWithInitCluster(numOfClusters=j, outputPath=op.join("..", ".."), numOfInitCluster=k)
                    v = KFoldValidation()
                    relt = v.validate(data, c1, foldNum=10)
                    
                    f = open(outputPath, "a")
                    f.write("clusterNum:" + str(j) + "\n")
                    f.write("segmentNum:" + str(k) + "\n")
                    f.write("foldNum:" + str(outputFolderNum) + "-" + str(outputFolderNum + 10) + "\n")
                    f.write("f1:" + str(relt["f1"]) + "\n")
                    f.write("---------------------------------------------------------\n")
                    f.write("precision:" + str(relt["precision"]) + "\n")
                    f.write("---------------------------------------------------------\n")
                    f.write("recall:" + str(relt["recall"]) + "\n")
                    f.write("---------------------------------------------------------\n")
                    f.write("accuracy:" + str(relt["accuracy"]) + "\n")
                    f.write("---------------------------------------------------------\n")
                    f.write("average_precision:" + str(relt["average_precision"]) + "\n")
                    f.write("---------------------------------------------------------\n")
                    f.write("roc_auc:" + str(relt["roc_auc"]) + "\n")
                    f.write("=========================================================\n")
                    f.close()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    logging.error("error happen clusterNum=" + str(j) + " initClusterNum=" + str(k))
                    logging.error(e.message)
                    continue
