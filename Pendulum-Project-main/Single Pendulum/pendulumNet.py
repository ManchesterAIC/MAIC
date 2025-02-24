import numpy as np
import math
import random 
import copy

def ReLU(inputs):
    for i in range(len(inputs)):
        inputs[i] = 0 if inputs[i] < 0 else inputs[i]
    return inputs

class PendulumNet:
    #Inputs = [Theta_, Theta]
    #Outputs = acceleration

    def __init__(self):
        
        #Architecture = [2,3,2,1]
        self.weight1 = np.random.rand(3, 2)
        self.bias1 = np.random.rand(3, 1)

        self.weight2 = np.random.rand(2, 3)
        self.bias2 = np.random.rand(2, 1)

        self.weight3 = np.random.rand(1, 2)
        self.bias3 = np.random.rand(1, 1)
       

    def forward(self, inputs):
        inputs = np.array(inputs)

        inputs = np.dot(self.weight1, inputs) + self.bias1
        inputs = inputs[0]
        inputs = ReLU(inputs)

        inputs = np.dot(self.weight2, inputs) + self.bias2
        inputs = inputs[0]
        inputs = ReLU(inputs)

        inputs = np.dot(self.weight3, inputs) + self.bias3
        inputs = inputs[0]
        
        return inputs[0]

    def reproduce(self, entropy = 5):
        child = copy.deepcopy(self)
        for parameter in [child.weight1, child.weight2, child.weight3,
                            child.bias1, child.bias2, child.bias3]:
            perturbation = np.random.randint(-5 * entropy, 5 * entropy, size=parameter.shape) / 100
            parameter += perturbation
        return child

net = PendulumNet()
print(net.forward([1,2]))