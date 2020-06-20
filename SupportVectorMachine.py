import numpy as np
import matplotlib.pyplot as plt
import pdb

X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

y = np.array([-1, -1, 1, 1, 1])
'''
epochs = 100000
w = np.zeros(len(X[0]))
learning_rate = 1


for epoch in range(1,epochs):
    for i, x in enumerate(X):
        if (y[i] * np.dot(X[i], w)) < 1:
            w = w + learning_rate*(X[i]*y[i]+(-2*(1/epoch)*w))

        else:
            w = w + learning_rate * (-2 * 1/epoch*w)



for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)


# Print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')
plt.show()

'''


class SupportVectorMachine:
    """
    example feature vector
    x = np.array([
    [x1,x2,x3,x4....xn]
    ])

    Ex y-vector:
    y ? np.array([1,1,1,-1,-1 ...-1])
    """
    def __init__(self, x:np.array, y:np.array)->None:


        self.x = x
        self.y = y

    """
    Initializes the w vector
    """
    def _get_init_w_vector(self):

        return np.zeros(len(self._get_x_vector()[0])) # Vector of weights which length should be minimized

    """
    Returns the training x_vector
    """
    def _get_x_vector(self)->np.array:

        return self.x
    """
    Returns the training x_vector
    """
    def _get_y_vector(self) -> np.array:
        return self.y

    """
    Trains the support vector machine
    """
    def train_alternative(self, learning_rate=1, epochs=100000,)->np.array:

        w = self._get_init_w_vector() # init w vector
        x = self._get_x_vector()
        y = self._get_y_vector()

        for epoch in range(1, epochs):
            for i, x_i in enumerate(x):
                w = w + self._get_gradient_vector_alternative(x_i, y[i], w, learning_rate, epoch)  # The SGD update

        return w

    def train(self, learning_rate=0.001, epochs=100000)->np.array:
        w = self._get_init_w_vector()
        x = self._get_x_vector()
        y = self._get_y_vector()

        for epoch in range(1,epochs):
            w -= learning_rate * self._get_gradient_vector(x, y, w, 10000)

        return w

    """
    Calculates the step to take using stochastic gradient decent
    """
    @staticmethod
    def _get_gradient_vector_alternative(x_sample, y_sample, w, learning_rate, epoch):

        # Max of 0 and  y_sample * np.dot(x_sample, w)
        if (y_sample * np.dot(x_sample, w)) < 1:
            step = (learning_rate * y_sample * x_sample - 2 * (1 / epoch) * w)
        else:
            step = learning_rate * (-2 * (1 / epoch) * w)

        return step

    @staticmethod
    def _get_gradient_vector(x_vectors, y_vectors, w, reg_strength = 10000):

        cost_max = 1 - y*(np.dot(x_vectors, w))
        dw = np.zeros(len(w))  # Empty w vector to fill
        for i, distance in enumerate(cost_max): # For each example, check which derivative to apply
            if max(0, distance) == 0:
                dw_contrib = w
            else:
                dw_contrib = w - (reg_strength*x_vectors[i]*y_vectors[i])
            dw += dw_contrib  # Increment the w vector by the contribution of that example

        return dw / len(y)  # return the average w vector

# Testing a simple SVM and plotting it


X = np.array([
    [-2,4,-1], # feature, feature, constant term
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

Y = np.array([-1, -1, 1, 1, 1])

test = SupportVectorMachine(X, Y)
#w_vector = test.train()
w_vector_two = test.train()
w_vector = test.train_alternative()

'''
# A line in R2 is described by ax + b*y + m = 0
a = w_vector[0]  # a coefficient in equation
b = w_vector[1]  # b coefficient in equation
m = w_vector[2]  # The constant in the equation

x_vector = np.linspace(-10,10) # Make an x vector
y_vector = (a*x_vector)/-b # The y is given by solving the equation for y
'''



for i, sample in enumerate(X):
    if i < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

        
x2=[w_vector[0],w_vector[1],-w_vector[1],w_vector[0]]
x3=[w_vector[0],w_vector[1],w_vector[1],-w_vector[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(x2, x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')




#plt.plot(x_vector, y_vector)
plt.show()

