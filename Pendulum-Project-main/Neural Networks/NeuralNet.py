import numpy as np

def relu(x):
    return max(0,x)

class NeuralNet:
    def __init__(self,  layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        for i in range(len(layers)-1):
            self.weights.append(np.random.rand(layers[i], layers[i+1]))
            self.biases.append(np.random.rand(layers[i+1], 1))
        
    def forward(self, inputs):
        #inputs = np.array(inputs).reshape(-1,1)
        for i in range(len(self.weights)):
            inputs = np.dot(self.weights[i], inputs) + self.biases[i]
            for input in inputs:
                input = relu(input)
        return inputs
    
    def forward(self, inputs, i):
        #custom forward function
        inputs = np.dot(self.weights[0], inputs) + self.biases[0]
        for input in inputs:
            input = relu(input)
        inputs = np.dot(self.weights[1], inputs) + self.biases[1]

        return inputs

            