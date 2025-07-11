import sys
import numpy as np
import matplotlib


# neuron with 4 inputs and a bias
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [0, 0, 0]

# neuron 1
for i in range(len(inputs)):
    output[0] += inputs[i] * weights1[i]

output[0] += bias1

# neuron 2
for i in range(len(inputs)):
    output[1] += inputs[i] * weights2[i]

output[1] += bias2

# neuron 3
for i in range(len(inputs)):
    output[2] += inputs[i] * weights3[i]

output[2] += bias3

print(output)