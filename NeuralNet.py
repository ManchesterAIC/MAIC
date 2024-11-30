import numpy as np

def step(x):
    return 1 if x > 0 else 0

class NeuralNet:
    def __init__(self,  layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        for i in range(len(layers)-1):
            self.weights.append(np.random.rand(layers[i], layers[i+1]))
            self.biases.append(np.random.rand(layers[i+1], 1))
        
    def forward(self, inputs):
        inputs = np.array(inputs).reshape(-1,1)

        for i in range(len(self.weights)):
            
            inputs = np.dot(self.weights[i].T, inputs) + self.biases[i]

            for input in inputs:
                input = step(input)

        return inputs

net = NeuralNet([5,2,3,2])
print(net.forward([5,4,3,2,1]))
            
