import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution = 0.02):

    # Setup the marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    colorMap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface (setting the boundries of the graph)
    x1_min = X[:, 0].min() - 1 # The minimum value for feature 0,  the sepal length,  of the setosa data
    x1_max = X[:, 0].max() + 1 # The maximum value for feature 0, the sepal length, of the versicolor data

    x2_min = X[:, 1].min() - 1 # The minimum value for feature 1, the petal length, of the versicolor data
    x2_max = X[:, 1].max() + 1 # The maximum value for feature 1, the petal length, of the versicolor data

    print("x1_min: " + str(x1_min))
    print("x1_max: " + str(x1_max))

    print("x2_min: " + str(x2_min))
    print("x2_max: " + str(x2_max))

    # Create a meshgrid, the aligning coordinates from two vectors, x1 and x2
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), # Return evenly spaces values given an interval
        np.arange(x2_min, x2_max, resolution)
    )

    # Set vector Z = to the class label predictions of the corresponding grid points
    Z = classifier.predict(
        np.array(
            [xx1.ravel(), xx2.ravel()]
        ).T
    )

    # Reshape the vector Z to be the same size (same number of columns)
    # as the two-feature matrix so that it can be
    # graphed & predicted just like the Iris training subset
    Z = Z.reshape(xx1.shape)

    # Graph the trained (predicted) data
    plt.contourf(
        xx1, xx2,
        Z,
        alpha = 0.3,
        cmap = colorMap
    )

    # Plot the limits of the datasets. Why? No clue.
    # Doesn't look like there's much of a difference with or w/o
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot the class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x = X[y == cl, 0], # Plot feature 1/0
            y = X[y == cl, 1], # Plot feature 2/1
            alpha = 0.8, # Transparency
            c = colors[idx],
            marker = markers[idx],
            label = cl,
            edgecolor = "black"
        )
