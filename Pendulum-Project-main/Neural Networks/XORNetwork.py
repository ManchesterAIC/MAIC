import Neuron

def step(x):
    return 1 if x > 0 else 0

class XORNetwork:
    def __init__(self):
        self.h1 = Neuron.Neuron([0.5,0.5], -0.45) 
        self.h2 = Neuron.Neuron([-0.5,-0.5], 1) 
        self.out = Neuron.Neuron([0.5,0.5], -0.9)
    
    def forward(self, inputs):
        return step(self.out.forward([step(self.h1.forward(inputs)), step(self.h2.forward(inputs))]))


#net = Neuron.Neuron([0.5,0.5], -0.45) #or Neuron
#net = Neuron.Neuron([0.5,0.5], -0.9)  #and Neuron
#net = Neuron.Neuron([-0.5, -0.5], 1)  #nand network

#net = XORNetwork()                    #XOR Network


print(net.forward([0,0])) #0
print(net.forward([1,0])) #1
print(net.forward([0,1])) #1
print(net.forward([1,1])) #0
