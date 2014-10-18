'''
Created on Oct 4, 2014

@author: merlin-teng
'''
from copy import deepcopy
from sklearn import cross_validation
from sklearn.metrics.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, average_precision_score, roc_auc_score
from util.myIO import DictionaryIO
from global_all.const_variable import Constants
import os.path as op
import numpy as np
import csv
import os
class KFoldValidation(object):
    '''
    class to execute K-Fold validation on a classifier
    '''


    def __init__(self, basePath, output = True):
        '''
        The basePath passed in will overide the output path of classifier
        '''
        self.output = output
        self.basePath = basePath
        if(op.exists(basePath) is not True):
            os.makedirs(basePath)
    
    def _output_validation_labels(self, predictLabel, trueLabel, train_idxes, test_idxes):
        dicIO = DictionaryIO()
        path = op.join(self.basePath, Constants.OUT_PUT_CONFIG_FILE_NAME)
        dicIO.loadFromFile(path)
        outputPath = op.join(self.basePath, str(dicIO.attr(Constants.CURRENT_RELT_NUMBER) - 1), "validation_labels.txt")
        f = open(outputPath, "a")
        f.write("predict labels:" + str(predictLabel) + "\n")
        f.write("true labels:" + str(trueLabel) + "\n")
        f.write("train idxes:"+(', '.join(map(str, train_idxes)))+"\n")
        f.write("test idxes:"+(', '.join(map(str, test_idxes)))+"\n")
        f.write("=========================================")
        f.close()
        
    def _output_validation_relt(self, relt):
        f = open(op.join(self.basePath, Constants.VALIDATION_RELT_NAME), "w")
        writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["f1","precision","recall","accuracy","average_precision","roc_auc"])
        for i in range(len(relt["f1"])):
            writer.writerow([relt["f1"][i],relt["precision"][i],relt["recall"][i],relt["accuracy"][i],relt["average_precision"][i],relt["roc_auc"][i]])
        writer.writerow(["mean"])
        writer.writerow([np.mean(relt["f1"]),np.mean(relt["precision"]),np.mean(relt["recall"]),np.mean(relt["accuracy"]),np.mean(relt["average_precision"]),np.mean(relt["roc_auc"])])
        writer.writerow(["std"])
        writer.writerow([np.std(relt["f1"]),np.std(relt["precision"]),np.std(relt["recall"]),np.std(relt["accuracy"]),np.std(relt["average_precision"]),np.std(relt["roc_auc"])])
        f.close()
        
    
    def validate(self, inputData, classifier, foldNum=10, stratified=False):
        data = inputData['data']
        labels = inputData['labels']
        if(stratified):
            kf = cross_validation.KFold(data.shape[0], n_folds=foldNum)
        else:
            kf = cross_validation.StratifiedKFold(y=labels, n_folds=foldNum)
        relt = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[], 'average_precision':[], 'roc_auc':[], 'models':[]}
         
        for train_idxes, test_idxes in kf:
            cf = deepcopy(classifier)
            cf.outputPath = self.basePath
            train_data, train_labels = data[train_idxes], labels[train_idxes]
            test_data, test_labels = data[test_idxes], labels[test_idxes]
            cf.fit(train_data, train_labels)
            predict_labels = cf.predict(test_data)
            print "#predict_labels:", predict_labels, "\n#true_labels:", test_labels, "\n"
            relt['accuracy'].append(accuracy_score(test_labels, predict_labels))
            relt['precision'].append(precision_score(test_labels, predict_labels))
            relt['recall'].append(recall_score(test_labels, predict_labels))
            relt['f1'].append(f1_score(test_labels, predict_labels))
            relt['average_precision'].append(average_precision_score(test_labels, predict_labels))
            relt['roc_auc'].append(roc_auc_score(test_labels, predict_labels))
            relt['models'].append(cf)
            if(self.output is True):
                self._output_validation_labels(predict_labels, test_labels, train_idxes, test_idxes)
        relt['best_model_idx'] = np.argmax(relt["f1"])
        self._output_validation_relt(relt)
        return relt
            
            
        
