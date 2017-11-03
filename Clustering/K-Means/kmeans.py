"""
K-Means clustering from scratch.
Developed with Zach Heick. GitHub: ZachHeick
---
Nov. 2016
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


def generate_data_2D(n_clust = 3):
    """
    Generate random two dimensional data
    ---
    KWargs:
        n_clust: how many clusters of data to generate
    """
    X1 = np.random.normal(-5,1,50).reshape(-1,1)
    y1 = np.random.normal(-5,1,50).reshape(-1,1)
    for _ in range(n_clust-1):
        X2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        y2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        X1 = np.vstack((X1,X2)).reshape(-1,1)
        y1 = np.vstack((y1,y2)).reshape(-1,1)
    X = np.hstack((X1,y1))
    return X


def generate_data_3D(n_clust = 3):
    """
    Generate random three dimensional data
    ---
    KWargs:
        n_clust: how many clusters of data to generate
    """
    X1 = np.random.normal(-5,1,50).reshape(-1,1)
    y1 = np.random.normal(-5,1,50).reshape(-1,1)
    z1 = np.random.normal(-5,1,50).reshape(-1,1)
    for _ in range(n_clust-1):
        X2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        y2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        z2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        X1 = np.vstack((X1,X2)).reshape(-1,1)
        y1 = np.vstack((y1,y2)).reshape(-1,1)
        z1 = np.vstack((z1,z2)).reshape(-1,1)
    X = np.hstack((X1,y1,z1))
    return X


class kmeans:

    def __init__(self, k = 5, random_seed=None, iters=1000):
        self._k = int(k)
        self._iters = iters
        if random_seed:
            np.random.seed(random_seed)

        self._any_changed = True

    def compute_distance_to_cluster_mean(self, clst, pt):
        """For a given point, calculate the distance to each centroid"""

    def get_clust_id_for_point(self,pt):
        pass

    def init_clusters(self, X):
        # To use the pre-built cluster sub-class you can use
        self.clusters = [self.cluster() for _ in range(0,self._k)]

        # Initialize random centroids
        # Max and Min of each dimension
        ranges = defaultdict(int)
        for col in range(0, X.shape[1]):
            ranges[col] = {}
            ranges[col]['min'] = min(X[:,col])
            ranges[col]['max'] = max(X[:,col])

        for k in range(0, self._k):
            centroid = []
            for col in range(0, X.shape[1]):
                centroid.append(np.random.uniform(ranges[col]['min'], ranges[col]['max']))
            self.clusters[k].mean = centroid


    def assign_cluster(self, X):
        """Find distance to each centroid and make cluster assignments"""
        for clstr in self.clusters:
            clstr.set_prev_members()

        for pt in X:
            dist = []
            for clstr in self.clusters:
                dist.append(np.linalg.norm(pt - clstr.mean))

            self.clusters[dist.index(min(dist))].add_member(pt)


    def update_centroid_locations(self):
        """Move centroid to the avg. location of its points"""
        changes = []
        for clstr in self.clusters:
            clstr.get_mean()
            changes.append(clstr.is_changed())
        self._any_changed = sum(changes)


    def fit(self, X):
        X = self.pandas_to_numpy(X)
        self.init_clusters(X)

        for iteration in range(self._iters):
            if self._any_changed:
                self.assign_cluster(X)
                self.update_centroid_locations()
            else:
                break

        self.training_data = X


    def predict(self, X):
        """Find distance to each centroid and make cluster assignments"""
        y = []
        for pt in X:
            dist = []
            for clstr in self.clusters:
                dist.append(np.linalg.norm(pt - clstr.mean))
            y.append(dist.index(min(dist)))
        return y


    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)

        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)


    def plot(self, X_new = None, figsize = (8,6)):
        """
        Plot the data.
        If centroids have been found, plot the centroids.
        If new data points are passed, make and plot predictions.
        ---
        KWargs:
            X_new: New points to assign
            figsize: 'figsize' argument to pass to matplotlib.pyplot.figure()
        """
        if X_new is not None:
            y_pred = self.predict(X_new)

        fig = plt.figure(figsize=figsize)
        X = self.training_data

        if X.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(xs = X[:,0],
                       ys = X[:,1],
                       zs = X[:,2],
                       c='b',
                       label='train',
                       s=10)

            centroids = [x.mean for x in self.clusters]
            x_c = [x[0] for x in centroids]
            y_c = [y[1] for y in centroids]
            z_c = [z[2] for z in centroids]

            ax.scatter(x_c,
                       y_c,
                       z_c,
                       s=40,
                       c='r',
                       label='centroid')

            if X_new is not None:
                ax.scatter(X_new[:,0],
                           X_new[:,1],
                           X_new[:,2],
                           c=y_pred,
                           label='predicts',
                           s=10)
            ax.legend(loc='lower right')

        elif X.shape[1] == 2:
            ax = fig.add_subplot(111)

            ax.scatter(X[:,0],
                       X[:,1],
                       c='b',
                       label='train',
                       s=10)

            centroids = [x.mean for x in self.clusters]
            x_c = [x[0] for x in centroids]
            y_c = [y[1] for y in centroids]

            ax.scatter(x_c,
                       y_c,
                       s=40,
                       c='r',
                       label='centroid')

            if X_new is not None:
                ax.scatter(X_new[:,0],
                           X_new[:,1],
                           c=y_pred,
                           label='predicts',
                           s=10)
            ax.legend(loc='lower right')



    class cluster:
        def __init__(self):
            """
            This sub-class stores all the information related to each cluster.
            mean: where is the average location of points in this cluster
            members: which data points are in this cluster
            prev_members: which data points were in this cluster last optimization step
            """
            self.mean = None # centroid
            self.members = []
            self.prev_members = []

        def set_prev_members(self):
            """
            Transfers current_members to prev_members for later comparison
            """
            self.prev_members = self.members
            self.members = []

        def add_member(self,pt):
            """
            Helper function to add a point to this cluster.
            ---
            Input: data point (array)
            """
            self.members.append(pt)

        def is_changed(self):
            """
            Checks if this cluster has been modified by the most recent
            optimizatino step.
            ---
            Output:
            did cluster change (bool)
            """
            return not np.array_equal(self.members,self.prev_members)

        def get_mean(self):
            """
            Given a list of the members, calculate the mean value in
            each dimension to give the mean location of the points.
            """
            means = []
            for dim in np.array(self.members).T:
                means.append(np.mean(dim))
            self.mean = means

        def get_total_square_distance(self):
            """
            Calculate the inertia for this cluster, by calculating
            the sum of squared distances to from the mean to all the
            points.
            """
            val = 0.
            for p in self.members:
                val += np.sqrt(np.sum((self.mean - p)**2))
            return val
