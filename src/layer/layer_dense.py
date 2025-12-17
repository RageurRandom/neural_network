import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) # in case of a dead network (outputs of 0), this should be changed

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases