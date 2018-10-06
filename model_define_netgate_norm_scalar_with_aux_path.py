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
        fc_in_dim = 6
        fc_out_dim = 256
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
        ##########  Aux Path Begin ###########
        ######################################

        ### ACC x        
        self.conv1_acc_x_aux = nn.Sequential(
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
        self.fc1_acc_x_aux = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())
        ### ACC y
        self.conv1_acc_y_aux = nn.Sequential(
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
        self.fc1_acc_y_aux = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())


        ### GYRO y        
        self.conv1_gyro_y_aux = nn.Sequential(
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
        self.fc1_gyro_y_aux = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())
        ### GYRO z        
        self.conv1_gyro_z_aux = nn.Sequential(
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
        self.fc1_gyro_z_aux = nn.Sequential(
            nn.Linear(fc_in_dim * out_channels, fc_out_dim),
            nn.ReLU())
        ### Netgate
        self.fc_all_aux = nn.Sequential(
            nn.Linear(fc_out_dim * 4, 256),
            nn.ReLU())

        self.fc_all_2_aux = nn.Linear(256, 4)
        ### Output
        self.fc_out_aux_acc_x = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_acc_y = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_acc_z = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_gyro_x = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_gyro_y = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_gyro_z = nn.Linear(fc_out_dim, 6)

        ######################################
        ##########  Aux Path End #############
        ######################################
        # self.conv2 = nn.Conv2d(4, , kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()


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
        '''
        self.fc_out_aux_acc_x = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_acc_y = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_acc_z = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_gyro_x = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_gyro_y = nn.Linear(fc_out_dim, 6)
        self.fc_out_aux_gyro_z = nn.Linear(fc_out_dim, 6)
        self.fc_out_dir = nn.Linear(fc_out_dim * 6, 6)
        '''
    def forward(self, x):
        acc_x_data = x[0]
        acc_y_data = x[1]
        acc_z_data = x[2]
        gyro_x_data = x[3]
        gyro_y_data = x[4]
        gyro_z_data = x[5]
        

        ### Acc X
        conv1_acc_x_data = self.conv1_acc_x(acc_x_data)
        fc1_acc_x_data = self.fc1_acc_x(conv1_acc_x_data.view(conv1_acc_x_data.size(0), -1))

        #print("size of X is",len(acc_x_data))
        #print("size of conv_X is",len(conv1_acc_x_data.view(conv1_acc_x_data.size(0), -1)))

        ### Acc Y
        conv1_acc_y_data = self.conv1_acc_y(acc_y_data)
        fc1_acc_y_data = self.fc1_acc_y(conv1_acc_y_data.view(conv1_acc_y_data.size(0), -1))
        ### Acc Z
        conv1_acc_z_data = self.conv1_acc_z(acc_z_data)
        fc1_acc_z_data = self.fc1_acc_z(conv1_acc_z_data.view(conv1_acc_z_data.size(0), -1))
        ### Gyro X
        conv1_gyro_x_data = self.conv1_gyro_x(gyro_x_data)
        fc1_gyro_x_data = self.fc1_gyro_x(conv1_gyro_x_data.view(conv1_gyro_x_data.size(0), -1))
        ### Gyro Y
        conv1_gyro_y_data = self.conv1_gyro_y(gyro_y_data)
        fc1_gyro_y_data = self.fc1_gyro_y(conv1_gyro_y_data.view(conv1_gyro_y_data.size(0), -1))
        ### Gyro Z
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


        #print('Before l2 norm fc_data', fc_data) 
        fc_data = F.normalize(fc_data , p=2, dim=1)
        fc_data = fc_data + 1 
        fc_data = fc_data * 2 
        #print('After l2 norm fc_data', fc_data)  
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



        ##################################################
        ############# Aux Paths ##########################
        ##################################################
        ### Axx X Aux
        conv1_acc_x_data_aux = self.conv1_acc_x(acc_x_data)
        fc1_acc_x_data_aux = self.fc1_acc_x(conv1_acc_x_data_aux.view(conv1_acc_x_data_aux.size(0), -1))

        #fc1_acc_x_data_aux = fc_data[:, 0:1] * fc1_acc_x_data_aux
        out_data_aux_acc_x = self.fc_out_aux_acc_x(fc1_acc_x_data_aux)

        ### Acc Y Aux
        conv1_acc_y_data_aux = self.conv1_acc_y(acc_y_data)
        fc1_acc_y_data_aux = self.fc1_acc_y(conv1_acc_y_data_aux.view(conv1_acc_y_data_aux.size(0), -1))
        #fc1_acc_y_data_aux = fc_data[:, 1:2] * fc1_acc_y_data_aux
        out_data_aux_acc_y = self.fc_out_aux_acc_y(fc1_acc_y_data_aux)

        ### Acc Z Aux
        conv1_acc_z_data_aux = self.conv1_acc_z(acc_z_data)
        fc1_acc_z_data_aux = self.fc1_acc_z(conv1_acc_z_data_aux.view(conv1_acc_z_data_aux.size(0), -1))
        #fc1_acc_z_data_aux = fc_data[:, 2:3] * fc1_acc_z_data_aux
        out_data_aux_acc_z = self.fc_out_aux_acc_z(fc1_acc_z_data_aux)

        ### Gyro X Aux
        conv1_gyro_x_data_aux = self.conv1_gyro_x(gyro_x_data)
        fc1_gyro_x_data_aux = self.fc1_gyro_x(conv1_gyro_x_data_aux.view(conv1_gyro_x_data_aux.size(0), -1))
        #fc1_gyro_x_data_aux = fc_data[:, 3:4] * fc1_gyro_x_data_aux
        out_data_aux_gyro_x = self.fc_out_aux_gyro_x(fc1_gyro_x_data_aux)

        ### Gyro Y Aux
        conv1_gyro_y_data_aux = self.conv1_gyro_y(gyro_y_data)
        fc1_gyro_y_data_aux = self.fc1_gyro_y(conv1_gyro_y_data_aux.view(conv1_gyro_y_data_aux.size(0), -1))
        #fc1_gyro_y_data_aux = fc_data[:, 4:5] * fc1_gyro_y_data_aux
        out_data_aux_gyro_y = self.fc_out_aux_gyro_y(fc1_gyro_y_data_aux)

        ### Gyro Z Aux
        conv1_gyro_z_data_aux = self.conv1_gyro_z(gyro_z_data)
        fc1_gyro_z_data_aux = self.fc1_gyro_z(conv1_gyro_z_data_aux.view(conv1_gyro_z_data_aux.size(0), -1))
        #fc1_gyro_z_data_aux = fc_data[:, 5:6] * fc1_gyro_z_data_aux
        out_data_aux_gyro_z = self.fc_out_aux_gyro_z(fc1_gyro_z_data_aux)

        return out_data, fc_data, out_data_aux_acc_x, out_data_aux_acc_y, out_data_aux_acc_z, out_data_aux_gyro_x, out_data_aux_gyro_y, out_data_aux_gyro_z
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func
