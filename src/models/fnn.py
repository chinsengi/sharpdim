import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

class FNN(nn.Module):
    def __init__(self, dataset, num_classes, nonlinearity, use_layer_norm=False, layers=[500,500,500]):
        super(FNN,self).__init__()
        self.num_classes = num_classes
        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        if dataset == 'fashionmnist':
            module_list = []
            if use_layer_norm:
                module_list.append(nn.LayerNorm(784,elementwise_affine=False))
            else:
                module_list.append(nn.Identity())
            module_list.append(nn.Linear(784,layers[0]))
            module_list.append(self.nonlinearity)
            for i in range(len(layers)-1):
                if use_layer_norm:
                    module_list.append(nn.LayerNorm(layers[i],elementwise_affine=False))
                else:
                    module_list.append(nn.Identity())
                module_list.append(nn.Linear(layers[i], layers[i+1]))
                module_list.append(self.nonlinearity)
            if use_layer_norm:
                module_list.append(nn.LayerNorm(layers[-1],elementwise_affine=False))
            else:
                module_list.append(nn.Identity())
            module_list.append(nn.Linear(layers[-1],num_classes))
            self.net = nn.Sequential(*module_list)
        elif dataset == 'cifar10':
            self.net = nn.Sequential(nn.Linear(3072,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,num_classes))
        elif dataset == '1dfunction':
            self.net = nn.Sequential(nn.Linear(1,20),
                            nn.ReLU(),
                            nn.Linear(20,20),
                            nn.ReLU(),
                            nn.Linear(20,1))

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        o = self.net(x)
        return o


def lenet():
    return LeNet()

def fnn(dataset, num_classes, nonlinearity, use_layer_norm=False):
    return FNN(dataset, num_classes, nonlinearity, use_layer_norm=use_layer_norm)

