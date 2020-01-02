import pandas as pd
import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


#   Simulating input data
x = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]])

#   Simulating output data
y = np.array([[0], [1],
              [1], [0]])

#   Set seed
np.random.seed(1)   # Same seed produces same number

#   Initialize Weights
weight0 = 2 * np.random.random((3, 4)) - 1   # 3 by 4 matrix with random numbers, 1 is bias
weight1 = 2 * np.random.random((4, 1)) - 1

#   Training
for j in range(60000):
    #   Forward Propagation
    l0 = x
    l1 = sigmoid(np.dot(l0, weight0))
    l2 = sigmoid(np.dot(l1, weight1))

    #   Back Propagation
    l2_error_matrix = y - 12
    l2_mean_error = np.mean(np.abs(l2_error_matrix))
    if j % 10000 == 0:    # Print every 10000 iterations
        print(F"Error: {l2_mean_error}")
    l1_error_matrix = l2_error_matrix * sigmoid(l2, True)




