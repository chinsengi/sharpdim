import os
import argparse
import json
import torch
import numpy as np

from src.trainer import train
from src.utils import load_net, load_data, eval_accuracy, save, savefig, use_gpu

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--gpuid',
                          default='0,', help='gpu id, [0] ')
    argparser.add_argument('--dataset', 
                          default='fashionmnist', help='dataset, [fashionmnist] | cifar10, 1dfunction')
    argparser.add_argument('--network', default='vgg',
                           help='network, [vgg] | fnn, resnet')
    argparser.add_argument('--num_classes', type=int, default=10)
    argparser.add_argument('--n_samples_per_class', type=int,
                           default=500, help='training set size, [1000]')
    argparser.add_argument('--optimizer', 
                           default='sgd', help='optimizer, [sgd]')
    argparser.add_argument('--n_iters', type=int,
                           default=10000, help='number of iteration used to train nets, [10000]')
    argparser.add_argument('--batch_size', type=int,
                           default=16, help='batch size, [100]')
    argparser.add_argument('--learning_rate', type=float,
                           default=1e-1, help='learning rate')
    argparser.add_argument('--momentum', type=float,
                           default='0.0', help='momentum, [0.0]')
    argparser.add_argument('--model_file', 
                           default='fashionmnist.pkl', help='filename to save the net, fashionmnist.pkl')
    argparser.add_argument('--train_size', type=int, default= 10000)
    args = argparser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    print('===> Config:')
    print(json.dumps(vars(args), indent=2))
    return args

def get_optimizer(net, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    else:
        raise ValueError('optimizer %s has not been supported'%(args.optimizer))

def main():
    args = get_args()

    device = use_gpu()
    criterion = torch.nn.MSELoss()
    train_loader, test_loader = load_data(args.dataset,
                                          args.train_size,
                                          batch_size=args.batch_size)
    net = load_net(args.network, args.dataset, args.num_classes)
    net = net.to(device)
    optimizer = get_optimizer(net, args)
    print(optimizer)

    print('===> Architecture:')
    print(net)

    print('===> Start training')
    dim_list = train(net, criterion, optimizer, train_loader, args, args.batch_size, args.n_iters, verbose=True)
    np.save('res/dim_list.npy', dim_list)

    train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)
    print('===> Solution: ')
    print('\t train loss: %.2e, acc: %.2f' % (train_loss, train_accuracy))
    print('\t test loss: %.2e, acc: %.2f' % (test_loss, test_accuracy))

    save(net, optimizer, 'res/', args.model_file)
    


if __name__ == '__main__':
    main()
