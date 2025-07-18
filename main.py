import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data

nnfs.init() # set random seed and default datatype in numpy

# this is a batch of inputs
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

X, y = spiral_data(100, 3)

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 22.2, -100]
output = []

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biaises = np.zeros((1, n_neurons)) # in case of a dead network (outputs of 0), this should be changed

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biaises


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(layer1.output)
print(activation1.output)