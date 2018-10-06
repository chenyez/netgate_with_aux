import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import parser_define

#import har_dataset_100_1000_1000 as har_dataset
import har_dataset_100_1000_1000_all_noise as har_dataset

import model_define_netgate_norm_scalar as model_define

def train_natural(model, train_loader, args):
    print('train_natural')

    model.train()
    for epoch in range(0, args.epochs):

        print('epochs', epoch)
        # pbar = tqdm.tqdm(total=len(train_loader.dataset))
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data_gpu = []
            for data_cpu in data:
                data_gpu.append(data_cpu.to(args.device))

            
            data = data_gpu
            # data = data.to(args.device)
            target = target.to(args.device)

            # print(data, label)

            # print('data', data.size())
            # print('label', label.size())

            output = model(data)
            # print(output)
            output = F.log_softmax(output, dim=1)


            # print('output', output)

            # print('data_x', data_x)
            # print('data_y', data_y)
            # print('output', output)
            loss = 0
            loss += model.loss_func(output, target)
            # loss += model.loss_func(F.log_softmax(model(data), dim=1), target)

            # print('loss', loss)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            if batch_idx % 100 == 0:
                print('loss', loss)
                # train_loss.append(loss)
                # test_loss.append(test(model, args.test_loader, args))
                # val_loss.append(test(model, args.val_loader, args))
            # pbar.update(len(data_x))
        # pbar.close()
        
        if epoch % 5 == 0:
            print('\n epoch {}'.format(epoch))
            # print('loss', loss)
            test(model, args.test_loader, args)

    # test(model, args.test_loader, args)

    # plt.plot(train_loss, 'g', label='train')
    # plt.plot(test_loss, 'b', label='test')
    # plt.plot(val_loss, 'r', label='validation')

    # plt.xlabel('iterations')
    # plt.ylabel('loss')
    # plt.ylim(0, 1)

    # plt.grid()
    # plt.savefig(args.dataset_name + '_' + args.model_save_name + '_h' + str(model.hidden_dim) + '_loss' + '.png')
    # plt.legend(loc='upper right')
    # # plt.show()
    # plt.clf()

def test(model, test_loader, args):
    model.eval()

    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
	data_gpu = []
	for data_cpu in data:
	    data_gpu.append(data_cpu.to(args.device))

	data = data_gpu
        target = target.to(args.device)

        output = model(data)
        output = F.log_softmax(output, dim=1)

        # test_loss += model.loss_func(output, target, size_average=False)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main_dataset():
    train_args = parser_define.train_args_define()
    
    torch.manual_seed(train_args.seed)
    train_args.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print('current device', train_args.device)
    
    # initial dataset
    har_train_dataset = har_dataset.HAR_Dataset(train=True,noise=True)
    har_trainloader = torch.utils.data.DataLoader(
        har_train_dataset,
        batch_size=train_args.train_batch_size, shuffle=True)
    
    har_test_dataset = har_dataset.HAR_Dataset(train=False,noise=True)
    har_testloader = torch.utils.data.DataLoader(
        har_test_dataset,
        batch_size=train_args.test_batch_size, shuffle=True)

    # model = model_define.Conv_Net_2()
    model = model_define.Conv_Net_6(
        kernel_size=32,
        stride=8,
        padding=0,
        out_channels=16)

    model_save_name = train_args.model_save_name
    model.to(train_args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)
    loss_func = F.nll_loss
    model.set_optimizer(optimizer)
    model.set_loss_func(loss_func)

    train_args.test_loader = har_testloader
    train_natural(model, har_trainloader, train_args)
    # test_data, test_label = har_dataloader.dataset[0]
    
    # print('test_data', test_data)
    # print('test_label', test_label)

    

    # output = conv_net_2(test_data)

    # print(output)


def main_data():

    # x_train = np.loadtxt('UCI HAR Dataset/train/X_train.txt')
    # y_train = np.loadtxt('UCI HAR Dataset/train/y_train.txt')

    body_acc_x_train = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt')
    body_gyro_x_train = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt')

    # print('mean = ', x_train[0].mean())
    # print('std = ', x_train[0].std())
    # print('max = ', x_train[0].max())
    # print('mean = ', x_train[0].min())


    # print('body_acc_x_train mean = ', body_acc_x_train.mean())
    body_acc_x_train = body_acc_x_train - body_acc_x_train.mean()
    body_acc_x_train /= body_acc_x_train.std()

    # print('body_acc_x_train mean = ', body_acc_x_train.mean())
    # print(body_acc_x_train)

    bias = 0
    data_num = 100

    x = np.arange(bias, 128 * data_num, 128)
    x_ori = np.arange(bias, 128 * data_num, 1)
    body_acc_x_train_mean = body_acc_x_train.mean(axis=1)
    plt.plot(x, body_acc_x_train_mean[bias:(bias + data_num)], alpha=0.5)
    plt.plot(x_ori, body_acc_x_train.flatten()[bias:(bias + data_num * 128)], alpha=0.5)
    plt.show()


    # bias = 0
    # data_num = 100

    # x = np.arange(bias, 128 * data_num, 128)
    # x_ori = np.arange(bias, 128 * data_num, 1)
    # body_gyrp_x_train = body_gyro_x_train.mean(axis=1)
    # plt.plot(x, body_gyrp_x_train[bias:(bias + data_num)], alpha=0.5)
    # plt.plot(x_ori, body_gyro_x_train.flatten()[bias:(bias + data_num * 128)], alpha=0.5)
    # plt.show()

if __name__ == '__main__':
    # main()
    #nn=nn.cuda()
    main_dataset()
