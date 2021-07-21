# Pytorch implementation of SNN with backpropagation which engenders Hebbian Learning
import os 
import matplotlib.pyplot as plt
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_mnist import download_mnist
import torch.nn.functional as F
from model import SpikingNet, SpikingNeuronLayer, NonSpikingNet, InputDataToSpikingLayer, OutputDataToSpikingLayer

bs = 1000
DATA_PATH = './data' 

# GPU if there
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else "cpu")

def train(model, device , train_set_loader, optimizer, epoch, logging_interval = 100):

    model.train()
    for batch_idx, (data, target) in enumerate(train_set_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % logging_interval == 0 :
            pred = output.max(1, keepdim = True)[1] # getting index of max log prob
            correct = pred.eq(target.view_as(pred)).float().mean().item()
            print("Train Epoch : {} [{}/{} ({:.0f}%)] Loss : {:.6f} Accuracy : {:.2f}%".format(
                epoch, batch_idx * len(data) , len(train_set_loader.dataset),
                100. * batch_idx/ len(train_set_loader), loss.item(), 
                100. * correct))
            
def train_many_epochs(model):
    epoch = 1
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    train(model, device, train_set_loader, optimizer, epoch, logging_interval=10)
    test(model, device, test_set_loader)

    epoch = 2
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    train(model, device, train_set_loader, optimizer, epoch, logging_interval=10)
    test(model, device, test_set_loader)

    epoch = 3
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train(model, device, train_set_loader, optimizer, epoch, logging_interval=10)
    test(model, device, test_set_loader)


def test(model, device, test_set_loader): 

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_set_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduce = True).item() # sum up batch loss
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_set_loader)
    print("")
    print("Test set : Average Loss : {:.4f}, Accuracy : {}/{} ({:.2f}%) ".format(
        test_loss,
        correct,len(test_set_loader.dataset),
        100. * correct / len(test_set_loader.dataset)))
    print("")



train_set,test_set = download_mnist(DATA_PATH)

train_set_loader = torch.utils.data.DataLoader(
        dataset = train_set,
        batch_size = bs,
        shuffle = True)

test_set_loader = torch.utils.data.DataLoader(
        dataset = test_set,
        batch_size = bs,
        shuffle = False)



spiking_model = SpikingNet(device = device, n_time_steps = 128, begin_eval = 0)
train_many_epochs(spiking_model)

non_spiking_model = NonSpikingNet().to(device)
train_many_epochs(non_spiking_model)


# Visualzing
data,target = test_set_loader.__iter__().__next()

# taking 1st testing example
x = torch.stack([data[0]])
y = target.data.numpy()[0]
plt.figure(figsize = (12,12))
plt.imshow(x.data.cpu().numpy()[0,0])
plt.title("Input image of x of label y = {}:".format(y))
plt.show()

# Plotting neuron's activations
spiking_model.visualize_all_neurons(x)
print("A hidden neuron that looks excited : ")
spiking_model.visualize_neuron(x,layer_idx = 0, neuron_idx = 0)
print("The output neuron of label : ")
spiking_model.visualize_neuron(x,layer_idx = 1, neuron_idx = y)
