import numpy as np

'''
Implementation of least squares linear regression using stochastic gradient descent
to find slope and intercept of the line.
'''


class LinearRegression:

    def __init__(self, x_vector, y_vector, learning_rate=0.1,training_cycles = 1000):

        self.x_vector = x_vector
        self.y_vector = y_vector
        self.learning_rate = learning_rate
        self.training_cycles = training_cycles
        self.predicted = []

    '''
    Return the vector of x values
    '''
    def _get_x_vector(self):

        return self.x_vector

    '''
    Return the vector of y values
    '''
    def _get_y_vector(self):

        return self.y_vector

    '''
    Return the learning rate of the classifier
    '''
    def get_learning_rate(self):

        return self.learning_rate

    '''
    Generates some random values for the slope and intercept to start with
    '''
    def _generate_random_start_values(self):

        return np.random.rand(), np.random.randn()

    '''
    Finds the slope and intercept by using stochastic gradient descent on the mean squared error function
    '''
    def stochastic_gradient_descent(self):

        x_vector = self._get_x_vector()
        slope, intercept = self._generate_random_start_values()  # Get some random start value for slope and intercept
        for _ in range(self.training_cycles):
            y_pred = self._get_predicted_y_vector(slope, intercept, x_vector)  # The predicted y value given current slope and intercept

            # Stochastic gradient descent means taking the gradient vector and walking in the negative direction of it
            sgd = np.array([slope, intercept])-self.get_learning_rate() * \
                       self.get_gradient_vector(y_pred, self._get_y_vector(), x_vector)
            slope = sgd[0]
            intercept = sgd[1]

        self.predicted = [slope,intercept]
        return np.array([slope, intercept])

    '''
    Calculates the vector of calculated y values for each x value in the data 
    '''
    def _get_predicted_y_vector(self, slope, intercept, x_vector):

        return slope*x_vector + intercept

    '''
    Calculates the gradient vector for a line equation
    '''
    def get_gradient_vector(self, y_pred, y_vector,x_vector):

        # Mean squared error derivative with respect to the slope is this:
        slope_gradient = -2/len(y_pred)*sum(x_vector*(y_vector-y_pred))

        # Mean squared error derivative with respect to intercept is this. (inner derivative of -intercept is 1)
        m_gradient = -2/len(y_pred)*sum(y_vector-y_pred)

        return np.array([slope_gradient,m_gradient])

    '''
    Getter for the trained slope and intercept
    '''
    def _get_predicted(self):

        return self.predicted

    '''
    Calcualates the means squared error of the trained classifier
    '''
    def get_error(self):

        slope, intercept = self._get_predicted() # Fetch the precicted intercept and slope
        mse = sum([val**2
                     for val in
                        (self._get_y_vector()-self._get_predicted_y_vector(slope,
                        intercept, self._get_x_vector()))]) \
                        / len(self._get_x_vector())

        return mse


''' 
Implementation of a line where x = y.
Should return a slope of 1 and intercept of 0
'''

x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])

test = LinearRegression(x, y)
print(test.stochastic_gradient_descent())
print(test.get_error())
