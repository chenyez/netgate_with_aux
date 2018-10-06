
import numpy as np

import torch
import torch.utils.data as data_utils
import torch.nn.functional as F


class HAR_Dataset(data_utils.Dataset):
    def __init__(self, train, noise):

        self.train = train

        if train == True:
            self.body_acc_x_train = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_acc_x_train.txt')
            self.body_acc_x_train = torch.from_numpy(self.body_acc_x_train).type(torch.FloatTensor)
            
            self.body_acc_y_train = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_acc_y_train.txt')
            self.body_acc_y_train = torch.from_numpy(self.body_acc_y_train).type(torch.FloatTensor)

            self.body_acc_z_train = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_acc_z_train.txt')
            self.body_acc_z_train = torch.from_numpy(self.body_acc_z_train).type(torch.FloatTensor)

            self.body_gyro_x_train = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_gyro_x_train.txt')
            self.body_gyro_x_train = torch.from_numpy(self.body_gyro_x_train).type(torch.FloatTensor)

            self.body_gyro_y_train = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_gyro_y_train.txt')
            self.body_gyro_y_train = torch.from_numpy(self.body_gyro_y_train).type(torch.FloatTensor)

            self.body_gyro_z_train = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_gyro_z_train.txt')
            self.body_gyro_z_train = torch.from_numpy(self.body_gyro_z_train).type(torch.FloatTensor)

            self.label = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/y_train.txt') - 1
            self.label = torch.from_numpy(self.label).type(torch.LongTensor)

        else:
            self.body_acc_x_test = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_acc_x_test.txt')
            self.body_acc_x_test = torch.from_numpy(self.body_acc_x_test).type(torch.FloatTensor)
            
            self.body_acc_y_test = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_acc_y_test.txt')
            self.body_acc_y_test = torch.from_numpy(self.body_acc_y_test).type(torch.FloatTensor)

            self.body_acc_z_test = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_acc_z_test.txt')
            self.body_acc_z_test = torch.from_numpy(self.body_acc_z_test).type(torch.FloatTensor)

            self.body_gyro_x_test = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_gyro_x_test.txt')
            self.body_gyro_x_test = torch.from_numpy(self.body_gyro_x_test).type(torch.FloatTensor)

            self.body_gyro_y_test = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_gyro_y_test.txt')
            self.body_gyro_y_test = torch.from_numpy(self.body_gyro_y_test).type(torch.FloatTensor)

            self.body_gyro_z_test = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/body_gyro_z_test.txt')
            self.body_gyro_z_test = torch.from_numpy(self.body_gyro_z_test).type(torch.FloatTensor)

            self.label = np.loadtxt('/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_yz_gyro_yz/ratio_100_std_1000_mean_1000_all_noise/y_test.txt') - 1
            self.label = torch.from_numpy(self.label).type(torch.LongTensor)


    def __getitem__(self, index):

        if self.train:
            return [self.body_acc_x_train[index].view(1, -1), self.body_acc_y_train[index].view(1, -1), self.body_acc_z_train[index].view(1, -1), self.body_gyro_x_train[index].view(1, -1), self.body_gyro_y_train[index].view(1, -1), self.body_gyro_z_train[index].view(1, -1)], self.label[index]
        else:
            return [self.body_acc_x_test[index].view(1, -1), self.body_acc_y_test[index].view(1, -1), self.body_acc_z_test[index].view(1, -1), self.body_gyro_x_test[index].view(1, -1), self.body_gyro_y_test[index].view(1, -1), self.body_gyro_z_test[index].view(1, -1)], self.label[index]
        # return [self.body_acc_x_train[index].reshape(1, 1, -1), self.body_acc_y_train[index].reshape(1, 1, -1)], self.label[index]

        # return 6 way data and a label

    def __len__(self):

        if self.train:
            return self.body_acc_x_train.size(0)
        else:
            return self.body_acc_x_test.size(0)
        # return rows of data
