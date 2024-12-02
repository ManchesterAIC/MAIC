import numpy as np

#Activation function that only returns 1 for positive inputs
def step(x):
    return 1 if x > 0 else 0

#Defines a feedworward neural network
class NeuralNet:
    def __init__(self,  layers):
        self.layers = layers
        self.weights = [] #Weight matrices
        self.biases = [] #Bias vectors

        #Weights and bias initialisation for each layer
        for i in range(len(layers)-1):
            self.weights.append(np.random.rand(layers[i], layers[i+1])) #Randomly generate weights
            self.biases.append(np.random.rand(layers[i+1], 1)) #Randomly generate biases
        
    #Performs a forward pass through the neural network
    def forward(self, inputs):
        #Converts inputs into a column vector
        inputs = np.array(inputs).reshape(-1,1)
        #Propagates inputs though each layer
        for i in range(len(self.weights)):
            #Adds biases to inputs and calculates their weighted sum
            inputs = np.dot(self.weights[i].T, inputs) + self.biases[i]
            #Apply activation function to each input
            for input in inputs:
                input = step(input)
        return inputs
#Initialise the neural network
net = NeuralNet([5,2,3,2])
print(net.forward([5,4,3,2,1]))
            
