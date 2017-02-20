"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rtl
import BagLearner as bl
import sys

def get_data(file_path):
    inf = open(file_path) 

    data = None
    if ('Istanbul.csv' in file_path):
        # We want to ignore the first column because it's a date field.
        data = np.array([map(float, s.strip().split(',')[1:]) for s in inf.readlines()[1:]])
    else:
        data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # Compute how much of the data is training and testing.
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # Separate out training and testing data.
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print trainX.shape
    print trainY.shape
    print testX.shape
    print testY.shape

    # Return all the data.
    return (trainX, trainY, testX, testY)

def evaluate_samples(learner, trainX, trainY, testX, testY):
    # Evaluate in sample.
    predY = learner.query(trainX) # Get the predictions.
    is_rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", is_rmse
    isc = np.corrcoef(predY, y=trainY)
    print "corr: ", isc[0,1]

    # Evaluate out of sample
    predY = learner.query(testX) # Get the predictions.
    os_rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", os_rmse
    osc = np.corrcoef(predY, y=testY)
    print "corr: ", osc[0,1]

    # Return error metric RMSE for graphing an overfit.
    return (is_rmse, os_rmse)

def problem_1():
    file_path = './Data/winequality-white.csv'

    # Get data.
    trainX, trainY, testX, testY = get_data(file_path)

    # Build a comma separated result of Leaf Size, in sample RMSE, out of sample RMSE.
    output = open('output-1.csv', 'w')
    output.write('Leaf Size,In Sample RMSE,Out of Sample RMSE\n')
    
    # What leaf size does overfitting occur?
    for i in range(1, 5001):
        print 'Leaf size: ' + str(i)
        # Create a learner and train it.
        learner = rtl.RTLearner(verbose = False, leaf_size = i)

        # Train the learner.
        learner.addEvidence(trainX, trainY)
        #print learner.author()

        # Test it and get error metrics.
        is_rmse, os_rmse = evaluate_samples(learner, trainX, trainY, testX, testY)

        # Append to output.
        output.write(str(i) + ',' + str(is_rmse) + ',' + str(os_rmse) + '\n')

    # Save output.
    output.close()    

def problem_2():
    file_path = './Data/winequality-red.csv'

    # Get data.
    trainX, trainY, testX, testY = get_data(file_path)

    # Build a comma separated result of Bag Size, in sample RMSE, out of sample RMSE.
    output = open('output-2.csv', 'w')
    output.write('Bag Size,In Sample RMSE,Out of Sample RMSE\n')
    
    # What bag size does overfitting occur?
    for i in range(1, 151):
        print 'Bag size: ' + str(i)
        # Create a learner and train it.
        learner = bl.BagLearner(learner = rtl.RTLearner, kwargs = {"leaf_size":1}, bags = i, boost = False, verbose = False)

        # Train the learner.
        learner.addEvidence(trainX, trainY)
        #print learner.author()

        # Test it and get error metrics.
        is_rmse, os_rmse = evaluate_samples(learner, trainX, trainY, testX, testY)

        # Append to output.
        output.write(str(i) + ',' + str(is_rmse) + ',' + str(os_rmse) + '\n')

    # Save output.
    output.close()   

if __name__=="__main__":
##    if len(sys.argv) != 2:
##        print "Usage: python testlearner.py <filename>"
##        sys.exit(1)
##    inf = open(sys.argv[1])
    # TODO test other datasets! also test the correct with by passing in arguments!!!
    #file_path = sys.argv[1] #'./Data/Istanbul.csv' 
    
    #problem_1()
    problem_2()
    #problem_3()
