import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
from decision_regions import plot_decision_regions

# Load the iris data as a pandas.DataFrame object
data = pd.read_csv("data/iris.data", header = None)

# Print the last 5 lines of the data
# (to check that it was loaded properly)
print(data.tail())

# Select the setosa and versicolor data set
y = data.iloc[0:100, 4].values # Column 4 is the name of the flower
y = np.where(y == "Iris-setosa", -1, 1) # Select the setosa and assign the class identifiers (class labels)
print(y)

# Extract the sepal and petal lengths (real traning data)
# Columns 0 and 2 are the sepal and petal, respectively
X = data.iloc[
    0:100,
    [0, 2]
].values

# Plot the setosa data
plt.scatter(
    X[:50, 0],
    X[:50, 1],
    color = "red",
    marker = "o",
    label = "setosa"
)

# Plot the versicolor data
plt.scatter(
    X[50:100, 0],
    X[50:100, 1],
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

""" -- Start training the data -- """

# Initialize the perceptron algorithm
perceptron = Perceptron(
    eta = 0.1,
    n_iter = 10
)

# Train the algorithm
perceptron.fit(X, y)

# Plot the errors over the epochs
plt.plot(
    range(1, len(perceptron.errors_) + 1),
    perceptron.errors_,
    marker = "o"
)

# Label the axes
plt.xlabel("Epochs")
plt.ylabel("Number of weight updates")

plt.show()

""" -- Display the decision regions -- """

# Plot the decision regions in order to see the data predictions (better)
plot_decision_regions(
    X, y, # The data
    classifier = perceptron
)

# Display the axes data
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend(loc = "upper left")

# Display the graph
plt.show()