import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,stride=1) # 28-5+1=24
        self.conv2 = nn.Conv2d(6,16,5,stride=1) # 12-5+1=8
        self.fc1 = nn.Linear(4*4*16,200)
        self.fc2 = nn.Linear(200,10)

    def forward(self,x):
        if x.ndimension()==3:
            x = x.unsqueeze(0)
        o = F.relu(self.conv1(x))
        o = F.avg_pool2d(o,2,2)

        o = F.relu(self.conv2(o))
        o = F.avg_pool2d(o,2,2)

        o = o.view(o.shape[0],-1)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o

class FNN(nn.Module):
    def __init__(self, dataset, num_classes, nonlinearity, use_layer_norm=False, layers=[500,500,500,84]):
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
            module_list.append(nn.Linear(784,layers[0]))
            module_list.append(self.nonlinearity)
            for i in range(len(layers)-1):
                if use_layer_norm:
                    module_list.append(nn.LayerNorm(layers[i],elementwise_affine=False))
                module_list.append(nn.Linear(layers[i], layers[i+1]))
                module_list.append(self.nonlinearity)
            if use_layer_norm:
                module_list.append(nn.LayerNorm(layers[-1],elementwise_affine=False))
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

