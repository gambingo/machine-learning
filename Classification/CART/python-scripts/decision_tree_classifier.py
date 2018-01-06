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
from collections import Counter
from collections import defaultdict




def grow_tree(data):
    """
    Grows a single decision tree.
    ---
    Args:
        data:   (array, dataframe) features as columns
    Returns:
         A dictionary (dict)
    """
    data = data_to_dict(data)



def data_to_dict(features, labels):
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
    dictionary['labels'] = np.array(labels)

    return dictionary


def best_feature(data, metric='entropy'):
    """
    Calculates the best feature to split on.
    ---
    Args:
        data:   (dict) output of data_to_dict
    KWargs:
        metric: (str) uncertainty metric, 'entropy' or 'gini'
    """
    # Initial Uncertainty
    if metric == 'entropy':
        initial_score = entropy(data['labels'])
    elif metric == 'gini':
        initial_score = gini(data['labels'])
    else:
        raise Exception("Valid uncertainty metrics are 'entropy' and 'gini'")

    # Information gain of each feature.
    best_gain = -1
    for feature in data.keys():
        if feature != 'labels':
            split_point, ent = best_split(data[feature], data['labels'],
                                          metric=metric)
            gain = initial_score - ent
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_split_point = split_point

    return best_gain, best_feature, best_split_point


def best_split(feature, labels, metric='entropy'):
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

        if metric == 'entropy':
            left_score = entropy(left)
            right_score = entropy(right)
        else:
            left_score = gini(left)
            right_score = gini(right)

        # Net Entropy
        net = (len(left)/len(labels))*left_score + \
            (len(right)/len(labels))*right_score

        if net < lowest_entropy:
            lowest_entropy = net
            split_point = test_point

    return split_point, lowest_entropy


# Both entropy and gini ignore zero probabilites
def entropy(labels):
    """Calculates the entropy for a given set of labels."""
    probs =  [freq/len(labels) for freq in Counter(labels).values()]
    return sum(-p*log(p, 2) for p in probs if p)


def gini(labels):
    """Calculates the gini score for a given set of labels"""
    probs =  [freq/len(labels) for freq in Counter(labels).values()]
    return sum(p*(1-p) for p in probs if p)
