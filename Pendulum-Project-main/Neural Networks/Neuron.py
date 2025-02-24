import numpy as np

def step(x):
    return 1 if x > 0 else 0

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
        return step(np.dot(self.weights, inputs) + self.bias)

