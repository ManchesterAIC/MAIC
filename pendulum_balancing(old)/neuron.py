import numpy as np

class Neuron:
    #Initialise weights and biases of an individual neuron
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
