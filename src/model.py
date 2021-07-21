# A simple SNN model implementation

import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

## In SNNs, A time axis exists and the NN sees data through time, also the activation
# functions are spikes instead of values and these spikes are raised past a certain 
# pre-activation threshold. 

# The pre-activation values constantly fades if the neurons aren't excited enough.(This
# is important as without fading, these past values may influence a spike when even it should
# be not there due to their additional support to incoming input activation value )

# The neurons after accumlating enough input activation through surpass the threshold causes
# them to "spike" and after this "spike" the neuron empties itself from it's activation and fires
# Once empty it goes to a "refactroy period" where the activation value is in -ve and this
# is here because it neuron (even biologically) takes a bit time before going into state of capable of
# firing again ( This is important as without it, if say another high input activation comes it will fire/spike
# instantaneously which rules out the whole biological imitation scenario )



class SpikingNeuronLayer(nn.Module):

    def __init__(self, device, n_inputs = 28*28, n_hidden = 100, decay_multiplier = 0.9, threshold=2.0, penalty_threshold = 2.5):
        super(SpikingNeuronLayer, self).__init__()
        self.device = device
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.decay_multiplier = decay_multiplier
        self.threshold = threshold
        self.penalty_threshold = penalty_threshold

        self.fc = nn.Linear(n_inputs, n_hidden)

        # initializations
        self.init_parameters()
        self.reset_state()
        self.to(self.device)

    def init_parameters():
        for param in self.parameters:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
    
    def reset_state(self):
        # Initialization/resetting the state for each neuron when starting predictions
        self.prev_inner = torch.zeros([self.n_hidden]).to(self.device)
        self.prev_outer = torch.zeros([self.n_hidden]).to(self.device)


    def forward(self, x):

        """ Forward step of neuron calling at every time step
        x : activated neurons below 
        return : a tuple of (state, output) for each time step, 
        Each tuple is having items which are of shape (batch_size, n_hidden)
        Total shape if casted : (2, batch_size, n_hidden)
        """
        if self.prev_inner.dim()==1 : 
            # Stacking a batch_size dimension after the reset_state()
            batch_size = x.shape[0]
            self.prev_inner = torch.stack(batch_size * [self.prev_inner])
            self.prev_outer = torch.stack(batch_size * [self.prev_outer])

        # Firstly, Weight matrix multiplied by input x 
        # As x was modified to flickering randomly through time or else x is 
        # already output of lower deep spiking layer
        input_excitation = self.fc(x)
        
        # Adding the result to decayed version of information we aready had at previous time step
        # The decay_multiplier serves the purpose of slowly fading the inner activation such that 
        # we don’t accumulate stimulis for too long to be able to have the neurons to rest.
        inner_excitation = input_excitation +  self.prev_inner * self.decay_multiplier
        
        # Computing activations of neuron to find it's output but before activation, we have 
        # a negative bias that refrain thing from firing too much. 
        outer_excitation = F.relu(inner_excitation - self.threshold)

        # If neuron fires then we need to reset it by subtracting it's activation to get it back to inner state 
        # plus a additional penalty for increasing the "refactory period"
        do_penalize_gate = (outer_excitation > 0).float()
        inner_excitation = inner_excitation - (self.penalty_threshold/self.threshold * inner_excitation) * do_penalize_gate

        # Setting internal values before returning
        delayed_return_state = self.prev_inner
        delayed_return_output = self.prev_outer
        self.prev_inner = inner_excitation
        self.prev_outer = outer_excitation

        # returning previous output
        return delayed_return_state,delayed_return_output



class InputDataToSpikingLayer(nn.Module):
    
    def __init__(self, device):
        super(InputDataToSpikingLayer, self).__init__()
        self.device = device
        
        self.to(self.device)
    
    def forward(self, x, is_2D = True):
        # Flattening 2d image to 1d for fc layer
        x = x.view(x.size(0),-1)
        random_activation_perceptron = torch.rand(x.shape).to(self.device)
        # attaching a random mask over the input x to have flickiring random input
        return random_activation_perceptron * x
        

class OutputDataToSpikingLayer(nn.Module):

    def __init__(self, average_output = True):

        """ average_output: might be needed if this is used within a regular neural net as a layer.
        Otherwise, sum may be numerically more stable for gradients with setting average_output=False.
        """
        super(OutputDataToSpikingLayer,self).__init__()
        if average_output : 
            self.reducer = lambda x, dim : x.sum(dim = dim)
        else:
            self.reducer = lambda x, dim : x.mean(dim = dim)

    def forward(self,x):
        if type(x)==list:
            x = torch.stack(x)
        return self.reducer(x,0)

class SpikingNet(nn.Module):

    def __init__(self, device, n_time_steps, begin_eval):
        super(SpikingNet, self).__init__()
        assert (0 <= begin_eval and begin_eval < n_time_steps)
        self.device = self.device
        self.n_time_steps = n_time_steps
        self.begin_eval = begin_eval

        self.input_conversion = InputDataToSpikingLayer(device)

        self.layer1 = SpikingNeuronLayer(
                device, n_inputs = 28 * 28 , n_hidden =100, decay_multiplier = 0.9, 
                threshold = 1.0, penalty_threshold = 1.5
                )

        self.layer2 = SpikingNeuronLayer(
                device, n_inputs = 100, n_hidden = 10, decay_multiplier = 0.9, 
                threshold = 1.0, penalty_threshold = 1.5
                )
        self.output_conversion = OutputDataToSpikingLayer(average_output = False) # sum on outputs

        self.to(self.device)

        def forward_through_time(self, x) :
        
        """ This acts as a layer. input and output is non-time related, All time iterations
        happen inside and returned layer is thus passed through global average pooling on 
        time axis before the return such as to be able to mix this pipeline with regular backprop
        layers such as the input data and the output data.
        """

        self.input_conversion.reset_state()
        self.layer1.reset_state()
        self.layer2.reset_state()

        out = []

        all_layer1_states = []
        all_layer1_outputs = []
        all_layer2_states = []
        all_layer2_outputs = []

        for _ in range(self.n_time_steps) : 
            xi = self.input_conversion(x)

            # For layer1 we take regular output
            layer1_state,layer1_output = self.layer1(xi)
            
            layer2_state, layer2_output = self.layer2(self.layer1_output)

            all_layer1_states.append(layer1_state)
            all_layer1_outputs.append(layer1_output)
            all_layer1_states.append(layer2_state)
            all_layer2_outputs.append(layer2_output)

            # we take inner state of layer2 because it's pre-activation and thus acts as out logits
            out.append(layer2_state)

        out = self.output_conversion(out[self.begin_eval : ])
        return out, [[all_layer1_states, all_layer1_outputs], [all_layer2_states, all_layer2_outputs]]

    def forward(self,x):
        out, _ = self.forward_through_time(x)
        return F.log_softmax(out, dim = -1)

    def visualize_all_neurons(self,x) : 
 
        assert x.shape[0] == 1 and len(x.shape) == 4, (
                "Pass only 1 example to SpikingNet.visualize(x)  with outer dim shape of 1")

        _, layers_state = self.forward_through_time(x)
        
        for i, (all_layer_states, all_layer_outputs) in enumerate(layers_state):
            layer_state = torch.stack(all_layer_states).data.cpu().numpy().squeeze().transpose()
            layer_output = torch.stack(all_layer_outputs).data.cpu().numpy().squeeze().transpose()

            self.plot_layer(layer_state, title="Inner state values of neurons for layer {}".format(i))
            self.plot_layer(layer_output, title="Output spikes (activation) values of neurons for layer {}".format(i))
       

    def visualize_neuron(self,x, layer_idx, neuron_idx):

        assert x.shape[0] == 1 and len(x.shape) == 4, (
                "Pass only 1 example to SpikingNet.visualize(x)  with outer dim shape of 1")

        _, layers_state = self.forward_through_time(x)

        all_layer_states, all_layer_outputs = layers_state[layer_idx]
        
        layer_state = torch.stack(all_layer_states).data.cpu().numpy().squeeze().transpose()
        layer_output = torch.stack(all_layer_outputs).data.cpu().numpy().squeeze().transpose()

        # Plot 'em
        self.plot_neuron(layer_state[neuron_idx], titile = "Inner state values of neuron {} of layer {}".format(neuron_idx,layer_idx))
        self.plot_neuron(layer_output[neuron_idx], title="Output spikes (activation) values of neuron {} of layer {}".format(neuron_idx, layer_idx))

    def plot_layer(self, layer_values, title) : 

        width = max(16, layer_values.shape[0]/8)
        height = max(4, layer_values.shape[1]/8)
        plt.figure(figsize = (width,height))
        plt.imshow(
                layer_values,
                interpolation = "nearest"
                cmap = plt.cm.rainbow
                )
        plt.title(title)
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("Neurons of layer")
        plt.show()

    def plot_neuron(self,neuron_through_time,title):

        width = max(16, neuron_through_time.shape/8)
        height = 4
        plt.figure(figsize = (width,height))

        plt.title(title)
        plt.plto(neuron_through_time)
        plt.xlabel("Time")
        plt.ylabel("Neurons's activation")
        plt.show()





## A  Non - Spiking Neural Network for comparison purposes


class NonSpikingNet(nn.Moduel):

    def __init(self):
        super(NonSpikingNet,self).__init__()
        self.layer1 = nn.Linear(28*28,100)
        self.layer2 = nn.Linear(100,10)

    def forward(self, x, is_2D = True):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return F.log_softmax(x, dim = -1)

