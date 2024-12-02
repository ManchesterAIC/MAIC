import Neuron

def step(x):
    return 1 if x > 0 else 0

class XORNetwork:
    #Initialise XOR (returns true when inputs are different) with weights and biases
    def __init__(self):
        self.h1 = Neuron.Neuron([0.5,0.5], -0.45) #First hidden neuron
        self.h2 = Neuron.Neuron([-0.5,-0.5], 1) #Second hidden neuron
        self.out = Neuron.Neuron([0.5,0.5], -0.9) #Output neuron
    
    #Forward pass through XOR network
    def forward(self, inputs):
        return step(self.out.forward([step(self.h1.forward(inputs)), step(self.h2.forward(inputs))]))

net = XORNetwork()

print(net.forward([0,0])) #0
print(net.forward([1,0])) #1
print(net.forward([0,1])) #1
print(net.forward([1,1])) #0

        