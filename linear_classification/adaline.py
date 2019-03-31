import numpy as np
from perceptron import Perceptron

class AdalineGradientDescent(Perceptron):
    """
    fit : Fit the training data
    ---------------
    vector X
        The training vectors
    vector y
        The target values
    """
    def fit(self, X, y):
        # Initialize the weight vector randomly
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(
            loc = 0.0,
            scale = 0.01,
            size = 1 + X.shape[1]
        )

        self.cost_ = [] # Initialize the cost vector

        # For number of epochs
        for i in range(self.n_iter):
            # Calculate the net input
            net_input = self.net_input(X)
            output = self.activation(net_input) # Set output equal to net_input (not necessary)

            errors = (y - output) # Update the errors (could be (y - net_input)

            """ begin gradient descent """
            self.w_[1:] += self.eta * X.T.dot(errors) # Update the weight vector
            
            # Sum of squared errors
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() # Calculate the cost of the function
            self.cost_.append(cost)
            """ end gradient descent """
        return self

    """
    activation : Calculate the linear activation
    ---------------
    """
    def activation(self, X):
        return X

    """
    predict : Predict the class labe
    ---------------
    This method is the same as the class' parent except it
    makes use of the activation() method
    """
    def predict(self, X):
        return np.where(
            self.activation(
                self.net_input(X) >= 0.0, 1, -1
            )
        )