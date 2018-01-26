"""
This is a custom random forest classification module. It is part of my work to
better understand algorithms by coding them from scratch.
---
J. Gambino
January, 2017
"""
import numpy as np
import pandas as pd
from math import sqrt
from warnings import warn
from scipy.stats import mode
from numpy.random import choice
from collections import defaultdict

from decision_tree_classifier import decision_tree_classifier as dtc


pd.options.mode.chained_assignment = None



class random_forest_classifier:
    """Build a random forest and use it to predict on new data"""

    def __init__(self, metric='entropy', n_trees=10, n_features=None,
                 min_uncertainty = 0):
        """
        KWargs:
            n_trees:    (int) Number of trees to grow
            n_features: (int) Number of features at each tree's disposal
        """
        self.metric = metric
        self.n_trees = n_trees
        self.n_features = n_features
        self.min_uncertainty = min_uncertainty


    def fit(self, features, labels):
        """
        Grow a random forest model. Grow decision trees with bootstrapped data
        and a random sample of features.
        ---
        Args:
            features:   (pandas dataframe) features as columns
            labels:     (pandas dataframe) labels as a single column
        """
        if features.shape[0] != len(labels):
            raise ValueError(('No. of feature rows ({}) does not equal the'
                              'number of labels ({}).'
                              ''.format(features.shape[0], len(labels))))

        self.trees = [dtc(metric=self.metric,
                          min_uncertainty=self.min_uncertainty)
                      for tree in range(self.n_trees)]

        if self.n_features is None:
            self.n_features = round(sqrt(features.shape[1]))

        for tree in self.trees:
            subset = self.sample_features(features)
            bag, oob = self.bootstrap(subset, labels)
            tree.fit(bag.drop('labels', axis=1), bag['labels'])


    def predict(self, features):
        """
        Ensemble the forest's predictions
        ---
        Args:
            features:   data of the same form provided to self.fit()
        """
        predictions = [tree.predict(features) for tree in self.trees]

        ensemble = np.array([])
        for ii in range(predictions[0].shape[0]):
            pred_ii = [x[ii] for x in predictions]
            ensemble = np.append(ensemble, mode(pred_ii)[0][0])

        return ensemble


    def sample_features(self, features):
        """
        Returns a subset of the feature data, with columns chosen at random
        ---
        Args:
            features:   (pandas dataframe) features as columns
        Returns:
            [unnamed]:  (pandas dataframe) a random subset of the data
        """
        if features.shape[1] <= self.n_features:
            msg = ('No. of available features ({}) is less than or equal to the'
                   ' number of requested sample features ({}).'
                   ''.format(features.shape[1], self.n_features))
            warn(msg)
            return features

        columns = choice(features.columns, self.n_features, replace=False)
        return features[columns]


    def bootstrap(self, features, labels):
        """
        Returns a bootstrapped aggregate or bagging of data and the out-of-bag
        samples.
        ---
        Args:
            Args:
                features:   (pandas dataframe) features as columns
                labels:     (pandas dataframe) labels as a single column
        Returns:
            bagged_data:    (dict) bootstrapped aggregate of data
            oob_data:       (dict) out of bag samples of data
        """
        if 'labels' in features.columns:
            raise NameError("No column in the features can be named 'labels'.")

        data = features
        data['labels'] = labels

        bag_mask = choice(data.index, data.shape[0], replace=True)
        oob_mask = [ind for ind in data.index if ind not in bag_mask]

        bag = data.loc[bag_mask]
        oob = data.loc[oob_mask]

        return bag, oob
