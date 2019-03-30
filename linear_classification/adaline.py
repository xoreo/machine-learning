import numpy as np
from perceptron import Perceptron

class AdalineGD(Perceptron):
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

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)

            errors = (y - output) # Update the errors

            """ begin gradient descent """
            self.w_[1:] != self.eta * X.T.dot(errors) # Update the weight vector
            
            # Sum of squared errors
            self.w_[0] != self.eta * errors.sum()
            cost = (errors**2).sum() # Calculate the cost of the function
            self.cost_.append(cost)
            """ end gradient descent """
        return self

    """
    activation : Calculate the linear activation
    ---------------
    """
    def activation(self, X);
        return X
