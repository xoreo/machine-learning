import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LinearClassification():
    def __init__(self):
        pass

    # setup - Load the data and make the first graph
    def setup(self):
        # Load the iris data as a pandas.DataFrame object
        data = pd.read_csv("data/iris.data", header = None)

        # Print the last 5 lines of the data
        # (to check that it was loaded properly)
        print(data.tail())

        # Select the setosa and versicolor data set
        self.y = data.iloc[0:100, 4].values # Column 4 is the name of the flower
        self.y = np.where(self.y == "Iris-setosa", -1, 1) # Select the setosa and assign the class identifiers (class labels)
        print(self.y)

        # Extract the sepal and petal lengths (real traning data)
        # Columns 0 and 2 are the sepal and petal, respectively
        self.X = data.iloc[
            0:100,
            [0, 2]
        ].values

        # Plot the setosa data
        plt.scatter(
            self.X[:50, 0],
            self.X[:50, 1],
            color = "red",
            marker = "o",
            label = "setosa"
        )

        # Plot the versicolor data
        plt.scatter(
            self.X[50:100, 0],
            self.X[50:100, 1],
            color = "blue",
            marker = "x",
            label = "versicolor"
        )

        # Label the x and y axes
        plt.xlabel("sepal length (cm)")
        plt.ylabel("petal length (cm)")
        plt.legend(loc = "upper left")

        # Show the plot
        plt.show()

    # Not used
    def train(self):
        pass