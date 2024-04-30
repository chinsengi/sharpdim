import torch.nn.init as init
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
        if len(x.shape) == 3:
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

class LeNet5(nn.Module):
    def __init__(self, output_dim, input_dim = None):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        init.xavier_normal_(self.conv1.weight.data)
        init.zeros_(self.conv1.bias.data)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        init.xavier_normal_(self.conv2.weight.data)
        init.zeros_(self.conv2.bias.data)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        init.xavier_normal_(self.fc1.weight.data)
        init.zeros_(self.fc1.bias.data)
        self.fc2 = nn.Linear(120, 84)
        init.xavier_normal_(self.fc2.weight.data)
        init.zeros_(self.fc2.bias.data)
        self.fc3 = nn.Linear(84, output_dim)
        init.xavier_normal_(self.fc3.weight.data)
        init.zeros_(self.fc3.bias.data)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def feature_layer(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x.view(-1, 84, 1)

    def classify(self, x):
        x = self.fc3(x)
        return x
    
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

def lenet5(num_classes):
    return LeNet5(num_classes)

def fnn(dataset, num_classes, nonlinearity, use_layer_norm=False):
    return FNN(dataset, num_classes, nonlinearity, use_layer_norm=use_layer_norm)

