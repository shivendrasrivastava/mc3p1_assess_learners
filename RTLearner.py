"""
Random tree learner.
"""

import numpy as np

class RTLearner(object):
    
    def __init__(self, leaf_size = 1, verbose = False):
        self.dataX = None
        self.dataY = None
        self.verbose = verbose
        # Helpers get called a lot more, so they need a second flag.
        self.helper_verbose = False
        self.leaf_size = leaf_size
        # -1 is used as the enum for representing a leaf node.
        self.LEAF = -1
        # When we initially get the tree, it won't be a matrix but a single list. Every 4 items are one row. It needs to be reshaped as so.
        self.column_count = 4

    def author(self):
        return 'msalim7' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner.
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        if self.verbose:
            print "RTLearner addEvidence(...)"

        # Merge new X values.
        self.dataX = self.mergeData(self.dataX, dataX)

        # Merge new Y values.
        self.dataY = self.mergeData(self.dataY, dataY)

        if self.verbose:
            print "X shape: " + str(self.dataX.shape)
            print "Y shape: " + str(self.dataY.shape)

        # Build tree with all X and Y values.
        self.random_tree = self.build_tree(self.dataX, self.dataY)

    def mergeData(self, oldData, newData):
        # Append new data to old data.
        return np.append(oldData, newData) if oldData else newData
        # TODO then prune out values that are not unique. np.unique doesn't seem to work as expected...
        #return np.unique(np.append(oldData, newData) if oldData else newData)
    
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        if self.verbose:
            print "RTLearner query(...)"
            print self.random_tree
            print 'Random tree shape: ' + str(self.random_tree.shape)
            #print 'Points: ' + str(points)
            print 'Points shape: ' + str(points.shape)

        # Get a value for each point given.
        results = []
        for i in range(0, points.shape[0]):
            results.append(self.query_helper(points[i]))

        if self.verbose:
            print 'Results length: ' + str(len(results))
            #print 'Results: ' + str(results)

        # Convert and return.
        return np.array(results)

    def query_helper(self, point):
        # Each entry of this decision tree in matrix format: [Factor, Split value, Left index, Right index]
        if self.verbose and self.helper_verbose:
            print "RTLearner query_helper(...)"
            
        # Start from row that represents "root" of the tree.
        i = 0

        # While the current row is not a leaf.
        while self.random_tree[i][0] != -1:
            # Get index of the factor we want to look at.
            factor = self.random_tree[i][0]

            # Get given value at that given factor index.
            value = point[factor]

            # Get the split value to decide to go left or right.
            split_value = self.random_tree[i][1]

            # Should we go left or right?
            if value <= split_value:
                # Go left.
                i = i + self.random_tree[i][2]
            else:
                # Go right.
                i = i + self.random_tree[i][3]

            # Are we about to go out of index?
            if self.verbose and self.helper_verbose and i >= self.random_tree.shape[0]:
                print "i = " + str(i) + ", tree shape = " + str(self.random_tree.shape)

        # Return (split) value of leaf.
        return self.random_tree[i][1]

    def build_tree(self, dataX, dataY):
        # Decision tree as a single list that need to be reshaped to a matrix.
        temp_tree =  self.build_tree_helper(dataX, dataY)

        # Figure out the reshape we need to do.
        row_count = temp_tree.shape[0] / self.column_count
        if self.verbose:
            print "Column count: " + str(self.column_count)
            print "Row count: " + str(row_count)

        # Reshape to 4xN where N is temp_tree.length/4.
        return temp_tree.reshape(row_count, self.column_count)

    def build_tree_helper(self, dataX, dataY):
        # Each entry of this decision tree in matrix format: [Factor, Split value, Left index, Right index]
        if self.verbose and self.helper_verbose:
            print "RTLearner build_tree(...)"
            
        # TODO check shape of both X andy Y?
        # Return leaf if:
        # 1. There is only one element left.
        # 2. There is less than or equal to the given leaf size elements.
        # 3. All remaining Y values are the same.
        if dataX.shape[0] == 1 or dataX.shape[0] <= self.leaf_size or all(dataY[0] == y for y in dataY):
            yMean = dataY.mean()
            if self.verbose and self.helper_verbose:
                print "Leaf value: " + str(yMean)
            return np.array([self.LEAF, yMean, None, None])

        # Determine random feature i to split on.
        i =  np.random.randint(dataX.shape[1])

        # Select 2 random values from ith feature.
        a = np.random.randint(dataX.shape[0])
        b = np.random.randint(dataX.shape[0])

        # Take avergage of the 2 randomly selected values.
        split_value = (dataX[a,i] + dataX[b,i]) / 2

        if self.verbose and self.helper_verbose:
            print "i: " + str(i)
            print "a: " + str(a)
            print "b: " + str(b)
            print "split_value: " + str(split_value)

        # Build left tree such that we pass it data that's less than or equal to the split value.
        li = dataX[:,i] <= split_value
        left_tree = self.build_tree_helper(dataX[li,:], dataY[li]) if True in li else [self.LEAF, dataY.mean(), None, None]

        # Build left tree such that we pass it data that's greater than the split value.
        ri = dataX[:,i] > split_value
        right_tree = self.build_tree_helper(dataX[ri,:], dataY[ri]) if True in ri else [self.LEAF, dataY.mean(), None, None]

        # Create root node. We divide by 4 because we get a single list inside of a list of lists.
        start_of_right_tree = len(left_tree) / self.column_count + 1
        root = np.array([i, split_value, 1, start_of_right_tree])

        if self.verbose and self.helper_verbose:
            print 'Root: ' + str(root)
            print 'Left: ' + str(left_tree)
            print 'Right: ' + str(right_tree)

        # Add left and right trees to root, then return the root.
        if left_tree != None:
            root = np.append(root, left_tree)
        if right_tree != None:
            root = np.append(root, right_tree)
        return root

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
