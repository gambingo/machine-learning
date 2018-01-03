import numpy as np
from math import log
from collections import Counter


def best_feature(data, labels, metric='entropy'):
    """
    Calculates the best feature to split on.
    ---
    Args:
        data:   (array, dataframe) features as columns
        labels: (list, array) labels for each data point
    KWargs:
        metric:     (str) uncertainty metric, 'entropy' or 'gini'
    """
    # TO-DO: Error Checking

    # Initial Uncertainty
    if metric == 'entropy':
        initial_score = entropy(labels)
    elif metric == 'gini':
        initial_score = gini(labels)
    else:
        raise Exception("Valid uncertainty metrics are 'entropy' and 'gini'")

    best_gain = -1
    for col in range(data.shape[1]):
        split_point, ent = best_split(data[:,col], labels, metric=metric)
        gain = initial_score - ent
        if gain > best_gain:
            best_gain = gain
            best_feature = col

    return best_feature, split_point, best_gain


def best_split(feature, labels, metric='entropy'):
    """
    Calculates the uncertainty of splitting on each data point and returns the
    best split point. Expects a single feature.
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
        raise Exception('Feature data and Labels must be the same length.')

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
