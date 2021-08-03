            ## VGG spiking variant model
# referenced paper -> https://openreview.net/forum?id=B1xSperKvH
import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Configuration : 'A' here is Average Pooling
cfg = {
        'VGG5' : [64,'A',128,128,'A'],
        'VGG9' : [64,'A',128,256,'A',256,512,'A',512,'A',512],
        'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
        'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
        'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
        'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class STDB(torch.autograd.Function):

    alpha = ''
    beta = ''

    @staticmethod
    def forward(ctx,input,last_spike):
        ctx.save_for_backward(last_spike)
        out = torch.zeros_like(input).to(device)
        out[input > 0 ] = 1.0 # spike 
        return out

    @staticmethod
    def backward(ctx,grad_output):
        last_spike, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = STDB.alpha * torch.exp(-1*last_spike)**STDB.beta
        return grad*grad_input, None


class LinearSpike(torch.autograd.Function):

    """piecewise-linear surrogate gradient 
        in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls dampening of piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        ctx.save_for_backward(last_spike)
        out = torch.zeros_like(input).to(device)
        out[input > 0 ] = 1.0 # spike
        return out

    @staticmethod
    def backward(ctx,grad_output):

        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = LinearSpike.gamma*F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad*grad_input, None

class PoissonGenerator(nn.Module):

    """The Poisson generator function outputs a Poisson spike train with rate proportional to the
    pixel value in the input. A random number is generated at every time-step for each pixel in the input
    image. The random number is compared with the normalized pixel value and if the random number
    is less than the pixel value an output spike is generated. This results in a Poisson spike train with
    rate equivalent to the pixel value if averaged over a long time
    """
    def __init__(self):
        super().__init__() 

    def forward(self,input):
        out = torch.le(torch.rand_like(input), torch.abs(input)*1.0).float()
        out = torch.mul(out, torch.sign(input))
        return out

class VGG_SNN(nn.Module):
    
    def __init__(self,vgg_name, activation = 'Linear', labels = 10, timesteps = 100, leak = 1.0, default_threshold = 1.0, alpha = 0.3, beta = 0.01, dropout = 0.2,
                kernel_size = 3, dataset = 'CIFAR10'):
        super().__init__()


        self.vgg_name = vgg_name

        # Activation function
        if activation == 'Linear':
            self.act_func = LinearSpike.apply
        elif activation == 'STDB' : # Spike time independent Backprop
            self.act_func = STDB.apply

        self.labels = labels
        self.timesteps = timesteps
        self.leak = torch.tensor(leak)
        STDB.alpha = alpha
        STDB.beta = beta

        self.dropout = dropout
        self.kernel_size = kernel_size
        self.dataset = dataset

        self.input_layer = PoissonGenerator()
        self.threshold = {}
        self.mem = {} # membrane potential
        self.mask = {} # dropout mask
        self.spike = {} # last spike time

        # layers
        self.features, self.classifier = self._make_layers(cfg[self.vgg_name])

        # intialization
        self._initialize_weights2()

        # threshold setup on conv layers 
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                self.threshold[l] = torch.tensor(default_threshold)

        prev = len(self.features)
        # threshold setup on linear layers for classifier part
        for l in range(len(self.classifier) -1):
            if isinstance(self.classifier[l], nn.Linear):
                self.threshold[prev + l] = torch.tensor(default_threshold)
            

        
    def _initialize_weights2(self):

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def _make_layers(self,cfg):

        # For features
        layers = []

        # Depending on the dataset 
        if self.dataset == 'MNIST' : 
            in_channels = 1
        else : 
            in_channels = 3

        for x in cfg:

            stride = 1
            if x == 'A':
                # drop the previous dropout layer
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size = 2, stride = 2)]
            else : 
                layers += [nn.Conv2d(in_channels,x, kernel_size = self.kernel_size, stride = stride, padding = (self.kernel_size - 1)//2, bias = False),
                            nn.ReLU(inplace = True)
                            ]
                layers += [nn.Dropout(self.dropout)]
                in_channels = x

        features = nn.Sequential(*layers)

        # For classifier
        layers = []

        if self.vgg_name == 'VGG5' and self.dataset != "MNIST" :

            layers += [nn.Linear(512*4*4,4096, bias = False)]
            layers += [nn.ReLU(inplace = True)]
            layers += [nn.Dropout(0.5)]

            layers += [nn.Linear(4096,4096, bias = False)]
            layers += [nn.ReLU(inplace = True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096,self.labels, bias = False)]

        elif self.vgg_name != 'VGG5' and self.dataset != "MNIST" :

            layers += [nn.Linear(512*2*2,4096, bias = False)]
            layers += [nn.ReLU(inplace = True)]
            layers += [nn.Dropout(0.5)]

            layers += [nn.Linear(4096,4096, bias = False)]
            layers += [nn.ReLU(inplace = True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096,self.labels, bias = False)]

        elif self.vgg_name == 'VGG5' and self.dataset == "MNIST" :

            layers += [nn.Linear(128*7*7,4096, bias = False)]
            layers += [nn.ReLU(inplace = True)]
            layers += [nn.Dropout(0.5)]

            layers += [nn.Linear(4096,4096, bias = False)]
            layers += [nn.ReLU(inplace = True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096,self.labels, bias = False)]

        elif self.vgg_name != 'VGG5' and self.dataset == "MNIST" :

            layers += [nn.Linear(512*1*1,4096, bias = False)]
            layers += [nn.ReLU(inplace = True)]
            layers += [nn.Dropout(0.5)]

            layers += [nn.Linear(4096,4096, bias = False)]
            layers += [nn.ReLU(inplace = True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096,self.labels, bias = False)]

        classifier = nn.Sequential(*layers)

        return features, classifier



    def threshold_update(self,scaling_factor = 1.0, thresholds = []):

        '''Function to update the threshold value for the conv layers and linear of classifier
        '''
        # Initialize thresholds
        self.scaling_factor = scaling_factor

        # Update on thresholds of conv layer in the feature part of the model
        for pos in range(len(self.features)) : 
            if isinstance(self.features[pos], nn.Conv2d):
                self.threshold[pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)

        prev = len(self.features)
        for pos in range(len(self.classifier)-1):
            if isinstance(self.classifier[pos], nn.Linear):
                self.threshold[prev+pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)



    def network_update(self,timesteps,leak):
        ''' Function to update the network's timesteps and leak value of spiking neurons
        '''
        self.timesteps = timesteps
        self.leak = torch.tensor(leak)

    def neuron_init(self,x):

        self.batch_size = x.size(0)
        self.width = x.size(2) 
        self.height = x.size(3)

        # Membrane potential, dropout mask and width,height init of the model's features
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                self.mem[l] = torch.zeros(self.batch_size,self.features[l].out_channels, self.width,self.height)

            elif isinstance(self.features[l], nn.Dropout):
                self.mask[l] = self.features[l](torch.ones(self.mem[l-2].shape)) # Conv->ReLu->Dropout, so use [l-2] to get conv2d shape

            elif isinstance(self.features[l], nn.AvgPool2d):
                self.width = self.width // self.features[l].kernel_size
                self.height = self.height // self.features[l].kernel_size

        prev = len(self.features)

        # Membrane potential, dropout mask init of the model's classifier part
        for l in range(len(self.classifier)-1):
            if isinstance(self.classifier[l],nn.Linear):
                self.mem[prev + l] = torch.zeros(self.batch_size, self.classifier[l].out_features)

            elif isinstance(self.classifier[l],nn.Dropout):
                self.mask[prev+l] = self.classifier[l](torch.ones(self.mem[prev + l -2].shape))
        

        # Spike time
        self.spike = copy.deepcopy(self.mem)
        for key,values in self.spike.items():
            for value in values : 
                value.fill_(-1000)

    def forward(self,x,find_max_mem = False,max_mem_layer = 0):

        self.neuron_init(x) 
        max_mem = 0.0

        for t in range(self.timesteps):
            out_prev = self.input_layer(x)

            for l in range(len(self.features)):
               
                if isinstance(self.features[l],nn.Conv2d):

                   if find_max_mem and l == max_mem_layer : 
                       if (self.features[l](out_prev)).max() > max_mem : 
                           max_mem = (self.features(out_prev)).max()
                       break
                    
                   mem_thr = (self.mem[l]/threshold[l]) - 1.0
                   out = self.act_func(mem_thr,(t-1-self.spike[l])) # the memory threshold and the last spike
                   rst = self.threshold[l]*(mem_thr>0).float()
                   
                   self.spike[l] = self.spike[l].masked_fill(out.bool(),t-1)
                   self.mem[l] = self.leak*self.mem[l] + self.features[l](out_prev) - rst
                   out_prev = out.clone()

                elif isinstance(self.features[l],nn.AvgPool2d):
                   out_prev = self.features[l](out_prev)

                elif isinstance(self.features[k],nn.Dropout):
                   out_prev = out_prev * self.mask[l]

            if find_max_mem and max_mem_layer < len(self.features):
                continue

            # Classifier
            out_prev = out_prev.reshape(self.batch_size, -1)
            prev = len(self.features)

            for l in range(len(self.classifier) -1):

                if isinstance(self.classifier[l], (nn.Linear)):

                    if find_max_mem and (prev+l)==max_mem_layer:
                        if(self.classifier[l](out_prev)).max() > max_mem : 
                            max_mem = (self.classifier[l](out_prev)).max()
                        break

                    mem_thr = (self.mem[prev+l]/self.threshold[prev+l]) -1.0
                    out = self.act_func(mem_thr, (t-1-self.spike[prev + l]))
                    rst = self.threshold[prev+l] * (mem_thr>0).float()

                    self.spike[prev+l] = self.spike[prev+l].masked_fill(out.bool(),t-1)
                    self.mem[prev+l] = self.leak*self.mem[prev+l] + self.classifier[l](out_prev) - rst
                    out_prev = out.clone()

                elif isinstance(self.classifier[l],nn.Dropout):
                    out_prev = out_prev * self.mask[prev+l]

                    
            # Compute classification layer outputs
            if not find_max_mem : 
                self.mem[prev+l+1] = self.mem[prev+l+1] + self.classifier[l+1](out_prev)
                
        if find_max_mem : 
            return max_mem 

        return self.mem[prev+l+1]

