"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rtl
import BagLearner as bl
import sys

if __name__=="__main__":
##    if len(sys.argv) != 2:
##        print "Usage: python testlearner.py <filename>"
##        sys.exit(1)
##    inf = open(sys.argv[1])
    
    # TODO test other datasets! also test the correct with by passing in arguments!!!
    #file_path = './Data/Istanbul.csv' #sys.argv[1]
    file_path = './Data/winequality-red.csv'
    #file_path = './Data/winequality-white.csv'

    inf = open(file_path) 

    data = None
    if ('Istanbul.csv' in file_path):
        # We want to ignore the first column because it's a date field.
        data = np.array([map(float, s.strip().split(',')[1:]) for s in inf.readlines()[1:]])
    else:
        data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print trainX.shape
    print trainY.shape
    print testX.shape
    print testY.shape

    # create a learner and train it
    #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    learner = rtl.RTLearner(verbose = True)
    #learner = rtl.RTLearner(verbose = True, leaf_size = 2)
    #learner = bl.BagLearner(learner = rtl.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)

    learner.addEvidence(trainX, trainY) # train it
    print learner.author()

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
