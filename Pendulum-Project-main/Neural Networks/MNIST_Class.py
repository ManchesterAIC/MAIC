import NeuralNet as nn
import torch
import torchvision
import json
import torchvision.transforms as transforms
import numpy 


#loading data
test_dataset = torchvision.datasets.MNIST(root="./data", transform=transforms.ToTensor(), 
                                                            train=False, download=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1, shuffle=False)

#Creating our Neural Net
MNIST_Class = nn.NeuralNet([784, 100, 10])


#Loading Parameters
with open("MNIST_Classification.json", 'r') as file:
    parameters = json.load(file)

MNIST_Class.weights[0] = numpy.array(parameters["l1.weight"])
MNIST_Class.weights[1] = numpy.array(parameters["l2.weight"])

MNIST_Class.biases[0] = numpy.array(parameters["l1.bias"])
MNIST_Class.biases[1] = numpy.array(parameters["l2.bias"])


#Testing Model
with torch.no_grad():
    n_correct = 0
    n_samples =0

    for images, labels in test_loader:

        #Data Processing from Torch.tensor --> Numpy Array
        images = images[0][0]
        temp = []

        for row in images:
            for pixel in row:
                temp.append(pixel)

        images = torch.tensor(temp).reshape(1, -1)
        images = images.numpy()[0]


        #Forward pass of our model
        outputs = MNIST_Class.forward(images, 1)
        
        n_samples += 1
        
        if outputs[labels[0]] == max(outputs):
            n_correct += 1
        
        acc = 100 * n_correct / n_samples
        print(f"accuracy = {acc}")