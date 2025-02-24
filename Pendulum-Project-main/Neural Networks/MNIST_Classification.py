import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

input_size = 784 
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                           transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,
                                           shuffle=True,)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,
                                           shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size,num_classes)
# loss and optmizer
criterion = nn.CrossEntropyLoss()
optmizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# training loop
n_total_steps = len(train_dataset)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 724
        images = images.reshape(-1,28*28) 
        
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backwards
        loss.backward()
        optmizer.step()
        optmizer.zero_grad()
        
        if(i%100) == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps} loss = {loss.item():.4f}")


# test
with torch.no_grad():
    n_correct = 0
    n_samples =0
    for images, labels in test_loader:
        images = images.reshape(-1,28*28)
        outputs = model(images)
        # value, index
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct = (predictions == labels).sum().item()
        
    acc = 100 * n_correct / n_samples
    print(f"accuracy = {acc}")

torch.set_printoptions(threshold=float('inf'))

import json
import numpy

parameters = {}
for name, param in model.named_parameters():
    parameters[name] = param.data.numpy().tolist()

with open("MNIST_Classification.json", "w") as file:
        json.dump(parameters, file, indent = 4)