import matplotlib.pyplot as plt

from perceptron import Perceptron
from decision_regions import plot_decision_regions
from linear_classification import LinearClassification

lc = LinearClassification()
lc.setup()

""" -- Start training the data -- """

# Initialize the perceptron algorithm
perceptron = Perceptron(
    eta = 0.1,
    n_iter = 100
)

# Train the algorithm
perceptron.fit(lc.X, lc.y)

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
    lc.X, lc.y, # The data
    classifier = perceptron
)

# Display the axes data
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend(loc = "upper left")

# Display the graph
plt.show()