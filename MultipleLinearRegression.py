import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn import linear_model
import pandas as pd

'''
Takes a list of numpy arrays as input vector. The last vector is presumed
to be the y vector
'''


class MultipleLinearRegression:

    def __init__(self, data_set, learning_rate=0.01):

        self.y_vector = data_set[-1]
        self.x_vectors = data_set[:-1]
        self.learning_rate = learning_rate
        self.predicted_slope = None
        self.predicted_intercept = None
        self.errors = []

    '''
    Initialize the coefficents to somethign random. Inlude the intercept at the end.
    '''
    def _get_random_slopes(self):
        return np.zeros((1, self._get_count_explanatory_variables()))

    @staticmethod
    def _get_random_intercept():

        return np.random.rand()

    def _get_learning_rate(self):

        return self.learning_rate

    def _get_count_explanatory_variables(self):

        return len(self.x_vectors)
    '''
    Gets the matrix of x_vectors
    '''
    def _get_x_vectors(self):

        return self.x_vectors

    '''
    Returns the output vector of the training data
    '''
    def _get_y_vector(self):

        return self.y_vector

    '''
    Predicts the y output based on a slope vector and intercept
    '''
    def _predict_y(self, slope_vector, intercept):

        y_pred = np.zeros((1, len(self.y_vector)))
        for i, x_vector in enumerate(self._get_x_vectors()):
            y_pred += slope_vector[0, i]*x_vector + intercept

        return y_pred

    '''
    The training method. Runs stochastic gradient descent 1000 times
    and updates the vectors of coefficients as well as the intercept
    '''
    def stochastic_gradient_descent(self):

        slope_vector, intercept = self._get_random_slopes(), self._get_random_intercept()

        for _ in range(1000):
            y_pred = self._predict_y(slope_vector, 0)
            self.calculate_error(y_pred)
            slope_vector -= self._get_learning_rate() * self._calculate_coefficient_gradient_vector(y_pred)
            intercept -=  self._get_learning_rate() * self._calculate_intercept_derivative(y_pred)

        self.predicted_slope, self.predicted_intercept = slope_vector, 0
        return slope_vector, intercept
    '''
    Calculates the MSE given current prediction values
    '''
    def calculate_error(self, y_pred):

        error = sum([val**2 for val in (self._get_y_vector() - y_pred)][0])
        self.errors.append(error)
    '''
    Calculates the gradient vector containing the derivatives of the MSE cost function with respect
    to each coefficient.
    '''
    def _calculate_coefficient_gradient_vector(self, y_pred):

        n = len(y_pred[0])
        gradient_vector = np.zeros((1, self._get_count_explanatory_variables()))
        for i, x_vector in enumerate(self._get_x_vectors()):
            gradient_vector[0, i] = -sum((x_vector * (self._get_y_vector() - y_pred)[0])) / n

        return gradient_vector
    '''
    Calculates the partial derivative of the cost with respect to 
    the intercept
    '''
    def _calculate_intercept_derivative(self, y_pred):
        n = len(y_pred[0])
        return sum((self._get_y_vector() - y_pred)[0]) / n

    '''
   Returns the trained coefficients and intercept
    '''
    def get_predicted_coefficients(self):

        return self.predicted_slope, self.predicted_intercept


# Fake some data in R3
x1 = np.array([1, 2, 3, 4, 5, 6])
x2 = np.array([0, 0, 0, 0, 0, 0])
y = np.array([1, 2, 3, 4, 5, 6])


classifier = MultipleLinearRegression(np.array([x1, x2, y]))
classifier.stochastic_gradient_descent()

# Print the coefficients
slope, intercept = classifier.get_predicted_coefficients()[0][0], classifier.get_predicted_coefficients()[1]
print(slope, intercept)

# Plot the error development in each sgd step
plt.plot([ _ for _ in range(0,len(classifier.errors))], classifier.errors)
plt.show()


# Benchmark using sklearn
df = pd.read_csv('./files/test_data.csv')
x = df[['x1','x2']]
y = df['y']
clf = linear_model.LinearRegression()
clf.fit(x, y)
print(clf.coef_)