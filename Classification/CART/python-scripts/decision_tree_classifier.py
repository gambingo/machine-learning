"""
This module is for growing a single decision tree. It is part of my work to
better understand algorithms by coding them from scratch.
---
J. Gambino
January, 2017
"""
import numpy as np
import pandas as pd
from math import log
from statistics import mode
from collections import Counter
from collections import defaultdict


class decision_tree_classifier:
    """Build a single decision tree and using it to predict on new data."""

    def __init__(self, metric='entropy'):
        """
        Args:
            metric: (str) 'entropy' or 'gini'
        """
        self.metric = metric


    def fit(self, features, labels):
        """
        Grows a single decision tree
        ---
        See grow_tree() docstring
        """
        self.tree = self.grow_tree(features, labels)


    def predict(self, features, labels=None):
        """Uses the tree to predict on new data"""
        data = self.data_to_dict(features, labels=labels)
        predictions = np.array([])

        num_points = len(data[list(data.keys())[0]])

        for ii in range(num_points):
            # A dictionary of feature values for a single data point
            data_point = defaultdict(float)
            for key in data.keys():
                data_point[key] = data[key][ii]

            predictions = np.append(predictions,
                                    self.classify(self.tree, data_point))
        return predictions


    def classify(self, sub_tree, data):
        """
        Classify a single data point using the provided tree
        ---
        Args:
            sub_tree:   (dict) a decision tree
            data_dict  ()
        Returns:
            class prediction or sub_tree
        """
        if isinstance(sub_tree, dict):
            if data[sub_tree['feature_name']] < sub_tree['split_point']:
                return self.classify(sub_tree['left'], data)
            else:
                return self.classify(sub_tree['right'], data)
        else:
            return sub_tree


    def grow_tree(self, features, labels):
        """
        Grows a single decision tree.
        ---
        Args:
            features:   (array, dataframe) features as columns
            labels:     (array, dataframe)
        Returns:
            tree:       (dict)
        """
        data = self.data_to_dict(features, labels=labels)
        return self.grow_branches(data)


    def data_to_dict(self, features, labels=None):
        """
        Transforms the input data into a dictionary. Decision trees need data in
        dictionary format so it can distinguish between features.
        ---
        Args:
            features:   (array, dataframe) features as columns
            labels:     (array, dataframe) labels as a single column
        Returns:    (dict)
        """
        dictionary = defaultdict(np.array)

        # Features
        if isinstance(features, pd.DataFrame):
            feature_names = list(features.columns)
            features = features.as_matrix()
            for col in range(features.shape[1]):
                dictionary[feature_names[col]] = features[:,col]

        elif isinstance(features, np.ndarray):
            for col in range(features.shape[1]):
                dictionary['feature' + str(col)] = features[:,col]

        else:
            msg = 'Input data must be a pandas dictionary or a numpy array.'
            raise TypeError(msg)

        # Labels
        if labels is not None:
            dictionary['labels'] = np.array(labels)

        return dictionary


    def grow_branches(self, data):
        """
        Creates a single node and returns two datasets split on that node.
        ---
        Args:
            data:   (array, dataframe) features as columns
        Returns:
             A dictionary (dict)
        """
        if self.uncertainty(data['labels']) == 0 or len(data.keys()) == 1:
            # If the labels are pure or all features have been exhausted
            return mode(data['labels'])

        gain, feature_name, split_point = self.best_feature(data)
        left, right = self.split(data, feature_name, split_point)
        return {'feature_name': feature_name,
                'gain': gain,
                'split_point': split_point,
                'left': self.grow_branches(left),
                'right': self.grow_branches(right)}


    def split(self, data, feature_name, split_point):
        """
        Splits the data into two sets. Removes the splitting feature from
        ---
        Args:
            data:           (dict) data in the format of data_to_dict output
            feature_name:   (str) name of feature to split on
            split_point:    (float) data point to split on
        Returns:
            left, right     (dict) input data split into two sets
        """
        left = defaultdict(np.array)
        right = defaultdict(np.array)

        mask = data[feature_name] < split_point
        for feature in data.keys():
            if feature != feature_name:
                left[feature] = data[feature][mask]
                right[feature] = data[feature][np.invert(mask)]

        return left, right


    def best_feature(self, data):
        """
        Calculates the best feature to split on.
        ---
        Args:
            data:   (dict) output of data_to_dict
        KWargs:
            metric: (str) uncertainty metric, 'entropy' or 'gini'
        """
        # Initial Uncertainty
        initial_score = self.uncertainty(data['labels'])

        # Information gain of each feature.
        best_gain = -1
        for feature in data.keys():
            if feature != 'labels':
                split_point, ent = self.best_split(data[feature],
                                                   data['labels'])
                gain = initial_score - ent
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split_point = split_point

        return best_gain, best_feature, best_split_point


    def best_split(self, feature, labels):
        """
        For a single feature, calculates the entropy (gini) of splitting on each
        data point and returns the best split point.
        ---
        Args:
            feature:    (list, array) data to split on
            labels:     (list, array) labels for each data point
        KWargs:
            metric:     (str) uncertainty metric, 'entropy' or 'gini'
        """
        # Error checking
        # TO-DO: Check shape as well
        if len(feature) != len(labels):
            raise Exception('Feature data and labels must be the same length.')

        lowest_entropy = 10
        for test_point in sorted(set(feature)):
            mask = [pt<test_point for pt in feature]
            left = labels[mask]
            right = labels[np.invert(mask)]

            left_score = self.uncertainty(left)
            right_score = self.uncertainty(right)

            # Net Uncertainty
            net = (len(left)/len(labels))*left_score + \
                (len(right)/len(labels))*right_score

            if net < lowest_entropy:
                lowest_entropy = net
                split_point = test_point

        return split_point, lowest_entropy


    def uncertainty(self, labels):
        """
        Calculates uncertainty using the specified metric.
        ---
        Args:
            labels: (ndarray) numpy array of label values
            metric: (str) 'entropy' or 'gini'
        """
        if self.metric == 'entropy':
            return self.entropy(labels)
        elif self.metric == 'gini':
            return self.gini(labels)
        else:
            raise Exception("Valid uncertainty metrics are 'entropy' and 'gini'")


    # Both entropy and gini ignore zero probabilites
    def entropy(self, labels):
        """Calculates the entropy for a given set of labels."""
        probs =  [freq/len(labels) for freq in Counter(labels).values()]
        return sum(-p*log(p, 2) for p in probs if p)


    def gini(self, labels):
        """Calculates the gini score for a given set of labels"""
        probs =  [freq/len(labels) for freq in Counter(labels).values()]
        return sum(p*(1-p) for p in probs if p)
