import sys
import numpy as np
import matplotlib


# neuron with 4 inputs and a bias
inputs = [1, 2, 3, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]

# output of one neuron : 
# weight1 * input1 + weight2 * input2 + ... + bias

# calculates the output of every neurons of this layer
layer_outputs = np.dot(weights, inputs) + biases


print(layer_outputs) 