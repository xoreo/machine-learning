import matplotlib.pyplot as plt
import numpy as np

from adaline import AdalineGradientDescent
from decision_regions import plot_decision_regions
from linear_classification import LinearClassification

lc = LinearClassification()
lc.setup()

""" -- Start training using Adaline -- """

# Create the subplots
figure, axes = plt.subplots(
    nrows = 1, ncols = 2,
    figsize = (10, 4)
)

# Initialize the algorithm for learning rate = 0.01
adaline1 = AdalineGradientDescent(
    n_iter = 10,
    eta = 0.01
)

# Train the algorithm
adaline1.fit(lc.X, lc.y)

# Plot the data
axes[0].plot(
    range(1, len(adaline1.cost_) + 1),
    np.log10(adaline1.cost_),
    marker = "o"
)

# Plot the axes
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("log(SSE)")
axes[0].set_title("Adaline @ eta = " + str(adaline1.eta))

# Initialize the algorithm for learning rate = 0.0001
adaline2 = AdalineGradientDescent(
    n_iter = 10,
    eta = 0.0001
)

# Run the algorithm
adaline2.fit(lc.X, lc.y)

# Plot the data
axes[1].plot(
    range(1, len(adaline2.cost_) + 1),
    adaline2.cost_,
    marker = "o"
)

# Plot the axes
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("SSE")
axes[1].set_title("Adaline @ eta = " + str(adaline2.eta))

# Show the plot
plt.show()