"""
A simple wrapper for bag learner.
"""
import numpy as np
import RTLearner as rt

class BagLearner(object):

    def __init__(self, learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
        # Create bag-number of learners.
        learners = []
        for i in range(0,bags):
            learners.append(learner(**kwargs))
        self.learners = learners

        # Set rest of class properties.
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        return 'msalim7' # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # For each bagged learner.
        for learner in learners:
            # TODO double check if this is what the N should be. What about N'?
            # Randomly select repeated data of size N (where N in this case is the entire training data set size).
            n = dataX.shape[0]
            ri = np.randint(n, n)
            randomX = dataX[ri,:]
            randomY = dataY[ri,:]
            learner.addEvidence(randomX, randomY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return -1

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
