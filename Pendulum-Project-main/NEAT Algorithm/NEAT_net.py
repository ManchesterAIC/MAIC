import math
import numpy as np
import random

class NEATNet:
    def __init__(self):

        self.layers = [4]
        for i in range(random.randint(1,10)):
            self.layers.append(random.randint(1,11))
        self.layers.append(1)

        self.layers = [4, 5, 5, 1]
        self.weights = []
        self.biases = []

        for i in range(len(self.layers)-1):
            self.weights.append(np.random.rand(self.layers[i+1], self.layers[i]))
            self.biases.append(np.random.rand(self.layers[i+1], 1))
    
    def normal(self, mu, sigma):
        return int(np.random.normal(mu, sigma))
    
    def ReLU(self, inputs):
        return np.maximum(0, inputs)

    def add_node(self):
        layer = random.randint(1, len(self.layers)-3)
        self.layers[layer] += 1

        new_weights = np.random.randint(1, 10, self.weights[layer][0].shape)
        self.weights[layer] = np.vstack((self.weights[layer], new_weights))

        new_biases = np.random.randint(1, 10, 1)/10
        self.biases[layer] = np.hstack((self.biases[layer], new_biases))
        
        new_weights = np.random.rand(self.weights[layer+1].shape[0], 1)
        self.weights[layer+1] = np.hstack((self.weights[layer+1], new_weights))
    
    def delete_node(self):
        layer = random.randint(1, len(self.layers)-3)
        while len(self.weights[layer]) <= 2:
            layer = random.randint(1, len(self.layers)-3)
        self.layers[layer] -= 1
        node = random.randint(0, len(self.weights[layer]) - 1)

        self.weights[layer] = np.delete(self.weights[layer], node, axis = 0)
        self.biases[layer] = np.delete(self.biases[layer], node)
        self.weights[layer+1] = np.delete(self.weights[layer+1], node, axis = 1)
    
    def forward(self, inputs):
        inputs = np.array(inputs)
        for i in range(len(self.weights)):
            inputs = np.dot(self.weights[i], inputs) + self.biases[i]
            inputs = inputs[0]
            inputs = self.ReLU(inputs)
        return inputs[0]

for i in range(100):
    net = NEATNet()
    print([net.weights[i].shape for i in range(len(net.weights))])
    print(net.forward([-3, 1,2,1]))
    net.delete_node()
    print([net.weights[i].shape for i in range(len(net.weights))])
    print(net.forward([-3, 1,2,1]))
    net.add_node()
    print([net.weights[i].shape for i in range(len(net.weights))])
    print(net.forward([-3, 1,2,1]))