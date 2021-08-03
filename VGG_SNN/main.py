import os, sys
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.utils
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torchviz import make_dot
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from model import VGG_SNN 

class AverageMeter(object) : 
    '''Computes and stores the average and current value
    ''' 
    def __init__(self, name, fmt = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    
    def update(self, val, n=1): 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmstr.format(**self.__dict__)


def train(epoch) : 
    ''' Train function
    '''
    global learning_rate
    
    # from the network_update method in our snn variant model
    model.module.network_update(timesteps = timesteps, leak = leak)
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    if epoch in lr_interval : 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19'])
    parser.add_argument('-lr','--learning_rate',    default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--timesteps',              default=100,                type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=0.7,                type=float,     help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha',                  default=0.3,                type=float,     help='parameter alpha for STDB')
    parser.add_argument('--beta',                   default=0.01,               type=float,     help='parameter beta for STDB')
    parser.add_argument('--optimizer',              default='Adam',             type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')    
    parser.add_argument('--momentum',               default=0.95,                type=float,     help='momentum parameter for the SGD optimizer')    
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.3,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches',      default=200,                type=int,       help='print training progress after this many batches')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Seed setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = args.architecture
    learning_rate       = args.learning_rate
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    lr_reduce           = args.lr_reduce
    timesteps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
    default_threshold   = args.default_threshold
    activation          = args.activation
    alpha               = args.alpha
    beta                = args.beta  
    optimizer           = args.optimizer
    weight_decay        = args.weight_decay
    momentum            = args.momentum
    amsgrad             = args.amsgrad
    dropout             = args.dropout
    kernel_size         = args.kernel_size
    test_acc_every_batch= args.test_acc_every_batch
    train_acc_batches   = args.train_acc_batches

    values = args.lr_interval.split()
    lr_interval = []
    for value in values : 
        lr_interval.append(int(float(value)*args.epochs))

    log_file = './logs/snn/'
    try : 
        os.mkdir(log_file)
    except OSError:
        pass

    # Log file name
    identifier = 'snn_' + architecture.lower() + '_' + dataset.lower() + "_" + str(timesteps)
    log_file += identifier + '.log'

    if args.log : 
        f = open(log_file,'w',buffering=1)
    else:
        f = sys.stdout


    f.write('\nRun on time : {}'.format(datetime.datetime.now()))

    f.write('\n\n Arguments : \n')

    for arg in vars(args):
        if args == 'lr_interval':
            f.write('\n\t {:20} : {}\n'.format(arg, lr_interval))
        else : 
            f.write('\n\t {:20} : {}\n'.format(arg,getattr(args,arg)))


    # Training setup

    if torch.cuda.is_available() and args.gpu : 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Dataset setup

    normalize = transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])

    if dataset in ["CIFAR10", "CIFAR100"]:
        transform_train = transforms.Compose([
                                transforms.RandomCrop(32,padding = 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize])

        transforms_test = transforms.Compose([transforms.ToTensor(),normalize])

    if dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root = '../src/data/', train=True, download = True, transform = transform_train)
        testset = datasets.CIFAR10(root = '../src/data',train = False, download = True, transform = transforms_test)
        labels = 10

    elif dataset == 'CIFAR100':
        trainset = datasets.CIFAR100(root = '../src/data', train = True, download = True, transform = transform_train)
        testset = datasets.CIFAR100(root='../src/data', train=False, download=True, transform = transform_test)
        labels = 100

    elif dataset == 'MNIST':
        trainset = datasets.MNIST(root='../src/data/MNIST/', train=True, download=True, transform=transforms.ToTensor())
        testset = datasets.MNIST(root='../src/data/MNIST/', train=False, download=True, transform=transforms.ToTensor())
        labels = 10

    # elif dataset == "IMAGENET":

    train_loader = DataLoader(trainset, batch_size = batch_size , shuffle = True)
    test_loader = DataLoader(testset, batch_size = batch_size, shuffle = False)

    if architecture[0:3].lower() == 'vgg':
        model = VGG_SNN(vgg_name = architecture, activation = activation, labels = labels, timesteps = timesteps,
                         leak = leak, default_threshold = default_threshold, alpha = alpha, beta = beta, dropout = dropout,
                         kernel_size = kernel_size, dataset = dataset)
    # elif architecture[0:3].lower() == 'res':

    ## Comment this following line, if you find key mismatch error and uncomment the DataParallel after the if block
    model = nn.DataParallel(model)

    if pretrained_snn : 

        state = torch.load(pretrained_snn, map_location = 'cpu')
        cur_dict = model.state_dict()

        for key in state['state_dict'].keys():

            if key in cur_dict : 
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data) 
                    f.write('\nLoaded {} from {}'.format(key,pretrained_snn))
                else : 
                    f.write('\nSize mismatch {}, size of loaded model {}, size of current model {}'.format(key, state['state_dict'][key].shape, model.state_dict()[key].shape))

            else : 
                f.write('\nLoaded weight {} not present in current model'.format(key))

        model.load_state_dict(cur_dict)

        if 'thresholds' in state.keys():
            try : 
                if stat['leak_mem']:
                    state['leak'] = state['leak_mem']
            except : 
                pass
            if state['timesteps'] != timesteps or state['leak'] != leak : 
                f.write('\n Timesteps/Leak mismatch between loaded SNN and current simulation timesteps/leak, current timesteps/leak {}/{},\
                    loaded timesteps/leak {}/{}'.format(timesteps, leak, state['timesteps'], state['leak']))

            thresholds = state['thresholds']
            model.module.threshold_update(scaling_factor = state['scaling_threshold'], thresholds = thresholds[:])
        else : 
            f.write('\n Loaded SNN model does not have thresholds')

    f.write('\n{}\n'.format(model))

    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    if optimizer == 'Adam': 
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, amsgrad = amsgrad, weight_decay = weight_decay)

    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay = weight_decay, momentum = momentum) 

    f.write('\n {}\n'.format(optimizer))

    max_accuracy = 0

    for epoch in range(1,epochs):
        start_time = datetime.datetime.now()
        if not args.test_only:
            train(epoch)
        test(epoch)

    f.write('\nHighest accuracy : {:.4f}'.format(max_accuracy))

