from math import log
from collections import Counter

def best_split(feature, labels, metric='entropy'):
    """
    Calculates the uncertainty of splitting on each data point and returns the
    best split point.
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

    if metric == 'entropy':
        initial_score = entropy(labels)
    elif metric == 'gini':
        initial_score = gini(labels)
    else:
        raise Exception("Valid uncertainty metrics are 'entropy' and 'gini'")

    best_gain = -1
    for test_point in sorted(set(feature)):
        mask = [pt<test_point for pt in feature]
        left = labels[mask]
        right = labels(np.invert(mask))

        if metric == 'entropy':
            left_score = entropy(left)
            right_score = entropy(right)
        else:
            left_score = gini(left)
            right_score = gini(right)

        # Net Entropy
        net = (len(left)/len(labels))*left + (len(right)/len(labels))*right

        # Information Gain
        gain = initial_score - net
        if gain > best_gain:
            best_gain = gain
            best_split = test_point

    return best_split


# Both entropy and gini ignore zero probabilites
def entropy(labels):
    """Calculates the entropy for a given set of labels."""
    probs =  [freq/len(labels) for freq in Counter(labels).values()]
    return sum(-p*log(p, len(probs)) for p in probs if p)


def gini(labels):
    """Calculates the gini score for a given set of labels"""
    probs =  [freq/len(labels) for freq in Counter(labels).values()]
    return sum(p*(1-p) for p in probs if p)
