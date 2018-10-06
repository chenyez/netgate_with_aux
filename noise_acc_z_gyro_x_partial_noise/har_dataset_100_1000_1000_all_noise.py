
import numpy as np

import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
import random

class HAR_Dataset(data_utils.Dataset):
    def __init__(self, train, noise):

        self.train = train
        self.noise = noise
        ratio = 100
        gaus_std = 1000
        mean = 1000
        if train == True:
            self.body_acc_x_train = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt')
            self.body_acc_y_train = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt')
            self.body_acc_z_train = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt')
            self.body_gyro_x_train = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt')
            self.body_gyro_y_train = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt')
            self.body_gyro_z_train = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt')
            self.label = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/train/Inertial Signals/y_train.txt') 

            if noise == True:
                length_of_data = len(self.body_acc_x_train)
                print("length of acc_x is:",length_of_data)
                for i in range(length_of_data):
                    a=random.randint(1,3)
                    if a==1:
                        print("training random clean")
                        self.body_acc_z_train[i] = self.body_acc_z_train[i]
                        self.body_gyro_x_train[i] = self.body_gyro_x_train[i]                                                  
                    else:
                        print("training random noise")
                        self.body_acc_z_train[i] = ratio * np.random.normal(mean,gaus_std,128)
                        self.body_gyro_x_train[i] = ratio * np.random.normal(mean,gaus_std,128) 


            #np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_y/body_acc_x_train.txt", self.body_acc_x_train, fmt='%.8f', delimiter=",",newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_acc_x_train.txt", self.body_acc_x_train, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_acc_y_train.txt", self.body_acc_y_train, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_acc_z_train.txt", self.body_acc_z_train, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_gyro_x_train.txt", self.body_gyro_x_train, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_gyro_y_train.txt", self.body_gyro_y_train, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_gyro_z_train.txt", self.body_gyro_z_train, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/y_train.txt", self.label, fmt='%i', newline='\n')


            self.body_acc_x_train = torch.from_numpy(self.body_acc_x_train).type(torch.FloatTensor)
            self.body_acc_y_train = torch.from_numpy(self.body_acc_y_train).type(torch.FloatTensor)
            self.body_acc_z_train = torch.from_numpy(self.body_acc_z_train).type(torch.FloatTensor)
            self.body_gyro_x_train = torch.from_numpy(self.body_gyro_x_train).type(torch.FloatTensor)
            self.body_gyro_y_train = torch.from_numpy(self.body_gyro_y_train).type(torch.FloatTensor)
            self.body_gyro_z_train = torch.from_numpy(self.body_gyro_z_train).type(torch.FloatTensor)
            self.label = torch.from_numpy(self.label).type(torch.LongTensor)

#np.savetxt(os.getcwd() + "/noise_rpm_spd_data/x_train_clean.csv", x_train, fmt='%.8f', delimiter=",",newline='\n')


        else:
            self.body_acc_x_test = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt')
            self.body_acc_y_test = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt')
            self.body_acc_z_test = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt')
            self.body_gyro_x_test = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt')
            self.body_gyro_y_test = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt')
            self.body_gyro_z_test = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt')
            self.label = np.loadtxt('/home/chenye/myung_netgate/Data/data/activity/UCI HAR Dataset/test/Inertial Signals/y_test.txt') 

            if noise == True:
                length_of_data = len(self.body_acc_x_test)
                print("length of acc_x is:",length_of_data)
                for i in range(length_of_data):
                    a=random.randint(1,3)
                    if a==1:
                        print("testing random clean")
                        self.body_acc_z_test[i] = self.body_acc_z_test[i]
                        self.body_gyro_x_test[i] = self.body_gyro_x_test[i]                                                  
                    else:
                        print("testing random noise")
                        self.body_acc_z_test[i] = ratio * np.random.normal(mean,gaus_std,128)
                        self.body_gyro_x_test[i] = ratio * np.random.normal(mean,gaus_std,128) 

            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_acc_x_test.txt", self.body_acc_x_test, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_acc_y_test.txt", self.body_acc_y_test, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_acc_z_test.txt", self.body_acc_z_test, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_gyro_x_test.txt", self.body_gyro_x_test, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_gyro_y_test.txt", self.body_gyro_y_test, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/body_gyro_z_test.txt", self.body_gyro_z_test, fmt='%.7e', delimiter=' ',newline='\n')
            np.savetxt("/home/chenye/myung_netgate/Data/chenye_pytorch_code/noise_acc_z_gyro_x_partial_noise/ratio_100_std_1000_mean_1000_all_noise/y_test.txt", self.label, fmt='%i', newline='\n')

            self.body_acc_x_test = torch.from_numpy(self.body_acc_x_test).type(torch.FloatTensor)
            self.body_acc_y_test = torch.from_numpy(self.body_acc_y_test).type(torch.FloatTensor)
            self.body_acc_z_test = torch.from_numpy(self.body_acc_z_test).type(torch.FloatTensor)
            self.body_gyro_x_test = torch.from_numpy(self.body_gyro_x_test).type(torch.FloatTensor)
            self.body_gyro_y_test = torch.from_numpy(self.body_gyro_y_test).type(torch.FloatTensor)
            self.body_gyro_z_test = torch.from_numpy(self.body_gyro_z_test).type(torch.FloatTensor)
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
