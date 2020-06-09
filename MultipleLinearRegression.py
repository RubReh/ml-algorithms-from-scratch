import numpy as np
import matplotlib.pyplot as plt
import pdb

'''
Takes a list of numpy arrays as input vector. The last vector is presumed
to be the y vector
'''

class MultipleLinearRegression:

    def __init__(self, data_set, learning_rate=0.0005):

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

    def _get_random_intercept(self):

        return np.random.rand()

    def _get_learning_rate(self):

        return self.learning_rate


    def _get_count_explanatory_variables(self):

        return len(self.x_vectors)

    def _get_x_vectors(self):

        return self.x_vectors

    def predict_y(self, slope_vector, intercept):

        y_pred = np.zeros((1, len(self.y_vector))) + intercept
        for i, x_vector in enumerate(self._get_x_vectors()):
            #pdb.set_trace()
            y_pred += slope_vector[0, i]*x_vector

        return y_pred

    def stochastic_gradient_descent(self):

        slope_vector, intercept = self._get_random_slopes(), self._get_random_intercept()
        y_pred = self.predict_y(slope_vector, intercept)
        self.calculate_error(y_pred)
        for _ in range(1000):
            #pdb.set_trace()
            slope_vector -= self._get_learning_rate() * self._calculate_gradient_vector(y_pred)
            intercept -=  self._get_learning_rate() * self._calculate_intercept_derivative(y_pred)

        self.predicted_slope, self.predicted_intercept = slope_vector, intercept
        return slope_vector, intercept

    def calculate_error(self, y_pred):

        error = sum([val**2 for val in (self._get_y_vector() - y_pred)])
        self.errors.append(error)





    def _calculate_gradient_vector(self, y_pred):

        n = len(y_pred[0])
        gradient_vector = np.zeros((1, self._get_count_explanatory_variables()))
        for i, x_vector in enumerate(self._get_x_vectors()):
            gradient_vector[0, i] = -sum((x_vector * (self._get_y_vector() - y_pred)[0])) / n

        return gradient_vector

    def _calculate_intercept_derivative(self, y_pred):
        n = len(y_pred[0])
        return sum((self._get_y_vector() - y_pred)[0]) / n

    def _get_y_vector(self):

        return self.y_vector

    def get_predicted(self):

        return self.predicted_slope, self.predicted_intercept









x1 = np.array([1, 2, 3, 4, 5, 6])
x2 = np.array([0, 0, 0, 0, 0, 0])
y = np.array([1, 2, 3, 4, 5, 6])


test = MultipleLinearRegression(np.array([x1, x2, y]))
test.stochastic_gradient_descent()
slope, intercept = test.get_predicted()[0][0], test.get_predicted()[1]
print(slope, intercept)



plt.plot([ _ for _ in range(0,len(test.errors))], test.errors)
plt.show()

