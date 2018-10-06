from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

from torchvision import datasets, transforms
from torch.autograd import Variable

class Conv_Net_6(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, out_channels=4):
        super(Conv_Net_6, self).__init__()
        self.conv1_acc_x = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv1_acc_y = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv1_acc_z = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv1_gyro_x = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv1_gyro_y = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv1_gyro_z = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        ######################################
        ##########  Aux Path  ################
        ######################################
        '''
        self.conv1_acc_x_aux = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv1_acc_y_aux = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv1_acc_z_aux = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1_acc_x_aux = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU(inplace=False)
        )
        
        self.fc1_acc_y_aux = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU(inplace=False))

        self.fc1_acc_z_aux = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU(inplace=False))
        '''
        ######################################
        ##########  Aux Path End #############
        ######################################
        # self.conv2 = nn.Conv2d(4, , kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()

        fc_in_dim = 6
        fc_out_dim = 256
        fc_data_cp=torch.FloatTensor()
        self.fc1_acc_x = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU()
        )
        
        self.fc1_acc_y = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())

        self.fc1_acc_z = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())

        self.fc1_gyro_x = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())

        self.fc1_gyro_y = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())

        self.fc1_gyro_z = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())

        self.fc_all = nn.Sequential(
            nn.Linear(fc_out_dim * 6, 256),
            nn.ReLU())

        self.fc_all_2 = nn.Linear(256, 6)

        self.fc_out = nn.Linear(fc_out_dim, 6)
        self.fc_out_dir = nn.Linear(fc_out_dim * 6, 6)
    
    def forward(self, x):
        acc_x_data = x[0]
        #acc_x_data = acc_x_data * 0
        acc_y_data = x[1]
        #acc_y_data = acc_y_data * 0
        acc_z_data = x[2]
        #acc_z_data = acc_z_data * 0
        gyro_x_data = x[3]
        #gyro_x_data = gyro_x_data * 0 
        gyro_y_data = x[4]
        #gyro_y_data = gyro_y_data * 0
        gyro_z_data = x[5]


        conv1_acc_x_data = self.conv1_acc_x(acc_x_data)
        # print(conv1_acc_x_data.size())
        #print("size of X is",len(acc_x_data))
        #print("size of conv_X is",len(conv1_acc_x_data.view(conv1_acc_x_data.size(0), -1)))

        fc1_acc_x_data = self.fc1_acc_x(conv1_acc_x_data.view(conv1_acc_x_data.size(0), -1))

        conv1_acc_y_data = self.conv1_acc_y(acc_y_data)
        fc1_acc_y_data = self.fc1_acc_y(conv1_acc_y_data.view(conv1_acc_y_data.size(0), -1))

        conv1_acc_z_data = self.conv1_acc_z(acc_z_data)
        fc1_acc_z_data = self.fc1_acc_z(conv1_acc_z_data.view(conv1_acc_z_data.size(0), -1))

        conv1_gyro_x_data = self.conv1_gyro_x(gyro_x_data)
        fc1_gyro_x_data = self.fc1_gyro_x(conv1_gyro_x_data.view(conv1_gyro_x_data.size(0), -1))

        conv1_gyro_y_data = self.conv1_gyro_y(gyro_y_data)
        fc1_gyro_y_data = self.fc1_gyro_y(conv1_gyro_y_data.view(conv1_gyro_y_data.size(0), -1))

        conv1_gyro_z_data = self.conv1_gyro_z(gyro_z_data)
        fc1_gyro_z_data = self.fc1_gyro_z(conv1_gyro_z_data.view(conv1_gyro_z_data.size(0), -1))
        
        # print(fc1_acc_y_data.size())
        cat_data = torch.cat((
            fc1_acc_x_data, 
            fc1_acc_y_data,
            fc1_acc_z_data,
            fc1_gyro_x_data,
            fc1_gyro_y_data,
            fc1_gyro_z_data), 
        dim=1)


        # print('cat_data', cat_data.size())
        fc_data = self.fc_all(cat_data)
        # print('fc_data', fc_data.size())
        fc_data = self.fc_all_2(fc_data)


        fc_data_cp = fc_data 
        #print('Before l2 norm fc_data', fc_data) 
        fc_data = F.normalize(fc_data , p=2, dim=1)
        '''     
        for i in range(len(fc_data)):
            
            #qn = torch.norm(fc_data[i], p=2, dim=0)
            #fc_data_clone = fc_data[i]/qn
            #fc_data[i] = fc_data_clone
        '''
        #print('After l2 norm fc_data', fc_data)  
        
        #fc_data = fc_data_cp
        #print("fc_Data is",fc_data)
        #fc_data = m(fc_data)

        fc_data = fc_data + 1 
        fc_data = fc_data * 2 
        m =nn.Softmax()
        fc_data = m(fc_data)
        #print('After softmax fc_data', fc_data)
        ##################################################
        ######### Netgate     ############################
        ##################################################
        
        fc1_w_acc_x_data = fc_data[:, 0].view(-1, 1) * fc1_acc_x_data
        fc1_w_acc_y_data = fc_data[:, 1].view(-1, 1) * fc1_acc_y_data
        fc1_w_acc_z_data = fc_data[:, 2].view(-1, 1) * fc1_acc_z_data
        fc1_w_gyro_x_data = fc_data[:, 3].view(-1, 1) * fc1_gyro_x_data
        fc1_w_gyro_y_data = fc_data[:, 4].view(-1, 1) * fc1_gyro_y_data
        fc1_w_gyro_z_data = fc_data[:, 5].view(-1, 1) * fc1_gyro_z_data
        
        #print("scalars are:",fc_data)
        add_out_data = fc1_w_acc_x_data + \
            fc1_w_acc_y_data + \
            fc1_w_acc_z_data + \
            fc1_w_gyro_x_data + \
            fc1_w_gyro_y_data + \
            fc1_w_gyro_z_data
        
        ##################################################
        ######### Netgate Ends ###########################
        ##################################################


        out_data = self.fc_out(add_out_data)

        # cat_w_data = torch.cat((
        #     fc1_w_acc_x_data, 
        #     fc1_w_acc_y_data,
        #     fc1_w_acc_z_data,
        #     fc1_w_gyro_x_data,
        #     fc1_w_gyro_y_data,
        #     fc1_w_gyro_z_data), 
        # dim=1)

        # out_data = self.fc_out_dir(cat_w_data)
        return out_data,fc_data,fc_data_cp
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func
