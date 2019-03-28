import numpy as np

# Perceptron - Linear classification algorithm
class Perceptron(object):
    """
    __init__ : Initialize the class variables
    ---------------
    float eta
        The learning rate (0.0 to 1.0). Some constant value.
    int n_iter
        The amount of iterations over the data set.
        Important for stopping the training when a dataset can not be classified linearly.
    int random_state
        The seed for a random number generator for the weight vector initialization.
    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        # Initialize the class variables
        self.eta = eta
        self.n_iter = n_iter # Also called the number of epochs (passes over the data set)
        self.random_state = random_state
    
    """
    fit : Fit the training data
    ---------------
    vector X
        The training vectors.
        vector shape: [n_samples, n_features]
        int n_samples is the number of samples.
        int n_features is the number of features

    vector y
        The target values (the actual, true values of the classification).
        vector shape: [n_samples]
        int n_samples is the number of samples.

    """
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state) # Seed the random number generator
        
        # Initialize the weight vector (every element is randomly chosen from rgen)
        # Standard deviation of 0.01, normally distributed, between 0 and 1
        self.w_ = rgen.normal(
            loc = 0.0,
            scale = 0.01,
            size = 1 + X.shape[1] # Add one for the bias column
        )

        self.errors_ = []

        # Run the algorithm n_iter times
        for i in range(self.n_iter):
            print("generation " + str(i))
            errors = 0 # The amount of incorrect predictions

            for xi, target in zip(X, y): # Iterate through the data
                # Implement the weight formula
                update = self.eta * (target - self.predict(xi)) # The updated weight
                self.w_[1:] += update * xi
                self.w_[0] += update
                # if update != 0.0:
                #     errors += int(update) # Update the errors
                errors += int(update != 0.0) # Update the errors
            self.errors_.append(errors)
        return self

    """
    net_input : Calculate the net input (the data adjusted with the weight)
    ---------------
    vector X
        The input data
    """
    def net_input(self, X):
        dot_product = np.dot(X, self.w_[1:]) # Get the dot product of the vectors
        return dot_product + self.w_[0] # Add the weight vector (bias)
    
    """
    predict : Return the class label after the unit step function
    ---------------
    vector X
        The input data
    """
    def predict(self, X):
        # Return 1 if self.net_input(X) >= 0, return -1 if self.net_input < 0
        return np.where(self.net_input(X) >= 0.0, 1, -1) # Predict which class the data is
