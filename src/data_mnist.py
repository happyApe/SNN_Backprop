import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms

# dataset chosen for now is mnist

def download_mnist(data_path) : 
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(1,0))])
    train_set = torchvision.datasets.MNIST(data_path, train=True,transform = transforms, download = True)
    test_set = torchvision.datasets.MNIST(data_path, train=False,transform = transforms, download = True)
    return train_set,test_set


