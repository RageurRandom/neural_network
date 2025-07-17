import sys
import numpy as np
import matplotlib

np.random.seed(0) # TODO remove

# this is a batch of inputs
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biaises = np.zeros((1, n_neurons)) # in case of a dead network (outputs of 0), this should be changed

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biaises


inputs_nb = len(X[0])
layer1 = LayerDense(inputs_nb, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)