import numpy as np
import matplotlib.pyplot as plt

'''
A kMeans classifier for R2
'''


class KmeansClassifier:


    def __init__(self, cluster_count, data):

        self.error = None
        self.cluster_count = cluster_count
        self.centroids = None
        self.data = data
        self.best_fit = None
        self.cluster_x, self.cluster_y = None, None

    '''
    Given the class's input data trains
    the centroids for the clusters. The index is updated 20 times 
    and 100 random initalized are attempted where the smallest error is picked
    '''

    def train(self, train_times=100):

        for i in range(train_times):

            self.centroids = self._randomize_centroids()
            for index in range(0, 20):
                self.set_cluster_means()

            error = self.calculate_error()
            if self.error is None:
                self._update_error_save_best_fit(error)
                continue

            if self.error > error:
                self._update_error_save_best_fit(error)

    '''
    Calculates the total distance from each data point to the nearest centroid.
    '''

    def calculate_error(self):

        return sum([self._get_error_distance(l, self.data) for l in range(0, self.cluster_count)])

    '''
    Sets classifiers error to given error argument and
    saves the best fit that has been found so far.
    '''

    def _update_error_save_best_fit(self, error):
        self.error = error
        self.cluster_x, self.cluster_y = self._get_merged_lists()
        self.best_fit = self.centroids.copy()

    '''
    Initalize the centroids to random positions between the highest and lowest x and y values 
    '''

    def _randomize_centroids(self):

        x_max, x_min, y_max, y_min = self.get_min_max()
        return [(np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)) for _ in
                range(0, self.cluster_count)]

    '''
    Finds the min and max x values for the datapoints in R2
    '''

    def get_min_max(self):

        x = self.data[:][0]
        y = self.data[:][1]

        return x.max(), x.min(), y.max(), y.min()

    '''
    Gets the minimum distance for a point to all cluster centroids
    '''

    def _get_min_distance_cluster(self, x1, y1):

        distances = [self._get_distance_square(x1, y1, x2, y2) for x2, y2 in self.centroids]

        return distances.index(min(distances))

    '''
    Gets the Euclidian distance between two points in R2
    '''

    def _get_distance_square(self, x1, y1, x2, y2):

        return pow((x1 - x2), 2) + pow((y1 - y2), 2)

    '''
    Get a list of all data points coupled with its nearest centroid
    '''

    def _classify_points(self):

        cluster_splits = [self._get_min_distance_cluster(x1, y1) for x1, y1 in self.data]

        return cluster_splits

    '''
    Classify all points and get a list of all x and y values coupled with their cluster
    '''

    def _get_merged_lists(self):
        cluster_with_x_values = []
        cluster_with_y_values = []
        classification = self._classify_points()

        for index in range(0, len(classification)):
            cluster_with_x_values.append((self.data[index][0], classification[index]))
            cluster_with_y_values.append((self.data[index][1], classification[index]))

        return cluster_with_x_values, cluster_with_y_values

    '''
    Calculate the total error for a certain cluster 
    '''

    def _get_error_distance(self, cluster, data_points):

        x2 = self.centroids[cluster][0]
        y2 = self.centroids[cluster][1]

        return sum([self._get_distance_square(x1, y1, x2, y2) for x1, y1 in data_points])

    '''
    Update the means of the cluster to be in the middle of the data points they've classified as
    '''

    def set_cluster_means(self):

        cluster_with_x_values, cluster_with_y_values = self._get_merged_lists()

        for cluster in range(0, k):
            values_in_cluster_x = [x for x, z in list(cluster_with_x_values)[:] if z == cluster]
            values_in_cluster_y = [y for y, z in list(cluster_with_y_values)[:] if z == cluster]
            self.centroids[cluster] = (self._mean(values_in_cluster_x), self._mean(values_in_cluster_y))

        return cluster_with_x_values, cluster_with_y_values

    '''
    Calculate the mean of a list
    '''

    def _mean(self, value_list):

        return sum(value_list) / len(value_list)

    '''
    Get the cluster labeled data points 
    '''

    def get_best_fit_data(self):

        return self.cluster_x, self.cluster_y

    '''
    Plot function only working for this example with two clusters in R2
    '''

    def plot(self):

        cluster_with_x_values, cluster_with_y_values = self.get_best_fit_data()
        values_in_cluster_x_1 = [x for x, z in list(cluster_with_x_values)[:] if z == 0]
        values_in_cluster_y_1 = [y for y, z in list(cluster_with_y_values)[:] if z == 0]

        values_in_cluster_x_2 = [x for x, z in list(cluster_with_x_values)[:] if z == 1]
        values_in_cluster_y_2 = [y for y, z in list(cluster_with_y_values)[:] if z == 1]

        plt.scatter(values_in_cluster_x_1, values_in_cluster_y_1, c='b', alpha=0.5)
        plt.scatter(values_in_cluster_x_2, values_in_cluster_y_2, c='y', alpha=0.5)
        plt.scatter(self.best_fit[0][0], self.best_fit[0][1], c='b', marker='v')
        plt.scatter(self.best_fit[1][0], self.best_fit[1][1], c='y', marker='v')
        plt.show()

    '''
    Print the total error of the classifier
    '''

    def print_error(self):

        print(self.error)


# Short implementation to test and plot a given data set with two distinct clusters. K = 2
X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1
x = X[:, 0]
y = X[:, 1]
k = 2

classifier = kMeansClassifier(k, X)
classifier.train()
classifier.plot()
classifier.print_error()
