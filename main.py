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
#import har_dataset
#import har_dataset_acc_x_noise as har_dataset
#import har_dataset_acc_x_noise_all_noise as har_dataset
#import har_dataset_gyro_x_noise_all_noise as har_dataset
#import har_dataset_acc_x_noise_partial_noise as har_dataset
#import har_dataset_acc_z_gyro_x_noise_all_noise as har_dataset
#import har_dataset_acc_x_gyro_z_noise_all_noise as har_dataset
#import har_dataset_acc_yz_gyro_xyz_noise_all_noise as har_dataset
#import har_dataset_acc_x_noise_all_noise as har_dataset
#import har_dataset_acc_x_noise_all_noise as har_dataset
#import model_define
#import model_define_non_netgate as model_define
import har_dataset_acc_z_gyro_x_noise_partial_noise as har_dataset
#import har_dataset_acc_yz_gyro_xyz_noise_partial_noise as har_dataset
import model_define_netgate_norm_scalar_with_aux_path as model_define

def train_natural(model, train_loader, args):
    print('train_natural')
    best_testing_accuracy = 0.0
    best_testing_accuracy_epoch = 0
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

            output, scalar_main, output_aux_acc_x, output_aux_acc_y, output_aux_acc_z, output_aux_gyro_x, output_aux_gyro_y, output_aux_gyro_z = model(data)
            # print(output)
            output = F.log_softmax(output, dim=1)
            output_aux_acc_x = F.log_softmax(output_aux_acc_x, dim=1)
            output_aux_acc_y = F.log_softmax(output_aux_acc_y, dim=1)
            output_aux_acc_z = F.log_softmax(output_aux_acc_z, dim=1)
            output_aux_gyro_x = F.log_softmax(output_aux_gyro_x, dim=1)
            output_aux_gyro_y = F.log_softmax(output_aux_gyro_y, dim=1)
            output_aux_gyro_z = F.log_softmax(output_aux_gyro_z, dim=1)

            # print('output', output)

            # print('data_x', data_x)
            # print('data_y', data_y)
            # print('output', output)
            loss = 0
            loss_aux_max = 0
            loss_aux_min = 0
            loss_main_model = 0
            loss_aux_acc_x = 0
            loss_aux_acc_y = 0
            loss_aux_acc_z = 0
            loss_aux_gyro_x = 0
            loss_aux_gyro_y = 0
            loss_aux_gyro_z = 0

            loss_main_model += model.loss_func(output, target)
            loss_aux_acc_x += model.loss_func(output_aux_acc_x, target)
            loss_aux_acc_y += model.loss_func(output_aux_acc_y, target)
            loss_aux_acc_z += model.loss_func(output_aux_acc_z, target)
            loss_aux_gyro_x += model.loss_func(output_aux_gyro_x, target)
            loss_aux_gyro_y += model.loss_func(output_aux_gyro_y, target)
            loss_aux_gyro_z += model.loss_func(output_aux_gyro_z, target)

            loss_aux_max = max(loss_aux_acc_x, loss_aux_acc_y, loss_aux_acc_z, loss_aux_gyro_x, loss_aux_gyro_y, loss_aux_gyro_z)
            loss_aux_min = min(loss_aux_acc_x, loss_aux_acc_y, loss_aux_acc_z, loss_aux_gyro_x, loss_aux_gyro_y, loss_aux_gyro_z)

            '''
            if loss_main_model>loss_aux_min:
                #loss+= loss_main_model + 2 * torch.abs(loss_main_model - loss_aux) + loss_aux
                #loss+= 5 * loss_main_model
                loss+= torch.abs( loss_main_model - loss_aux_min ) + 5* loss_aux_min
            else:
               
                loss+= loss_main_model + 5* loss_aux_min
            '''
            loss+= 5 * loss_main_model + loss_aux_acc_x + loss_aux_acc_y +loss_aux_acc_z +loss_aux_gyro_x +loss_aux_gyro_y +loss_aux_gyro_z


            # print('loss', loss)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            if batch_idx % 100 == 0:
                print('Total loss, main loss', loss, loss_main_model)
                print('Aux losses', loss_aux_acc_x, loss_aux_acc_y, loss_aux_acc_z, loss_aux_gyro_x, loss_aux_gyro_y, loss_aux_gyro_z)
                
                # train_loss.append(loss)
                # test_loss.append(test(model, args.test_loader, args))
                # val_loss.append(test(model, args.val_loader, args))
            # pbar.update(len(data_x))
        # pbar.close()
        
        if epoch % 5 == 0:
            print('\n epoch {}'.format(epoch))
            print('\n training scalar main is:\n',scalar_main)
            #print('\n training scalar main is:',scalar_main)
            # print('loss', loss)
            testing_accuracy = test(model, args.test_loader, args, epoch)
            if testing_accuracy > best_testing_accuracy  :
                best_testing_accuracy = testing_accuracy
                best_testing_accuracy_epoch = epoch
    print("best testing accuracy is:",best_testing_accuracy,best_testing_accuracy_epoch)
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

def test(model, test_loader, args, epoch):
    model.eval()

    test_loss_main = 0
    test_loss_aux_acc_x = 0
    test_loss_aux_acc_y = 0
    test_loss_aux_acc_z = 0
    test_loss_aux_gyro_x = 0
    test_loss_aux_gyro_y = 0
    test_loss_aux_gyro_z = 0
    test_loss_aux_max = 0
    test_loss_aux_min = 0
    correct = 0

    correct_aux_acc_x = 0
    correct_aux_acc_y = 0
    correct_aux_acc_z = 0
    correct_aux_gyro_x = 0
    correct_aux_gyro_y = 0
    correct_aux_gyro_z = 0

    plt.close()
    fig= plt.figure(figsize=(30, 5))
    test_accuracy_list = torch.zeros(6)
    test_target_list = torch.zeros(6)
    test_iter = 0
    for data, target in test_loader:
        test_iter+=1
	data_gpu = []
        sca_all = []
        sca_all_aux = []
	for data_cpu in data:
	    data_gpu.append(data_cpu.to(args.device))

	data = data_gpu
        target = target.to(args.device)

        output, scalar_main, output_aux_acc_x, output_aux_acc_y, output_aux_acc_z, output_aux_gyro_x, output_aux_gyro_y, output_aux_gyro_z = model(data)
        output = F.log_softmax(output, dim=1)
        output_aux_acc_x = F.log_softmax(output_aux_acc_x, dim=1)
        output_aux_acc_y = F.log_softmax(output_aux_acc_y, dim=1)
        output_aux_acc_z = F.log_softmax(output_aux_acc_z, dim=1)
        output_aux_gyro_x = F.log_softmax(output_aux_gyro_x, dim=1)
        output_aux_gyro_y = F.log_softmax(output_aux_gyro_y, dim=1)
        output_aux_gyro_z = F.log_softmax(output_aux_gyro_z, dim=1)

        # select scalar according to targe equals to 0
        #scalar_main = scalar_main[target == 0]

        print(scalar_main.size())

        ## Calculate mean value for main scalar
        '''
        #sca_0 = torch.mean(scalar_main[:,0])

        for i in range(6):
            fig.add_subplot(1, 6, i + 1)
            plt.hist(scalar_main[:,i].detach(),
                bins=50, 
		alpha=0.3, 
		label=str(i))
	    
            plt.legend(loc=1)

        # plt.ylim([0, 200])
	# plt.legend(loc=1)
        plt.draw()
        plt.pause(1)
        plt.cla()
	'''

        # select scalar according to targe equals to 0
        
	select_input = 0

        ## Calculate mean value for main scalar
        #sca_0 = torch.mean(scalar_main[:,0])
        '''
        for i in range(6):
            fig.add_subplot(1, 6, i + 1)
            scalar_select = scalar_main[target == i]
            plt.hist(scalar_select[:, select_input].detach(),
                bins=50, 
		alpha=0.3, 
		label=str(i))
	    
            plt.legend(loc=1)

        # plt.ylim([0, 200])
	# plt.legend(loc=1)
        plt.draw()
        filename = str(epoch)
        plt.savefig('/home/chenye/Documents/Testing/All_clean_acc_x_to_all_target/' + 'fig_' + filename + '.png')
        plt.pause(1)
        plt.cla()
        '''
        
        sca_1 = torch.mean(scalar_main[:,1])
        sca_2 = torch.mean(scalar_main[:,2])
        sca_3 = torch.mean(scalar_main[:,3])
        sca_4 = torch.mean(scalar_main[:,4])
        sca_5 = torch.mean(scalar_main[:,5])
        #sca_all.append([sca_0,sca_1,sca_2,sca_3,sca_4,sca_5])


        loss_test = 0

        test_loss_main += model.loss_func(output, target, size_average=False)
        test_loss_aux_acc_x += model.loss_func(output_aux_acc_x, target, size_average=False)
        test_loss_aux_acc_y += model.loss_func(output_aux_acc_y, target, size_average=False)
        test_loss_aux_acc_z += model.loss_func(output_aux_acc_z, target, size_average=False)
        test_loss_aux_gyro_x += model.loss_func(output_aux_gyro_x, target, size_average=False)
        test_loss_aux_gyro_y += model.loss_func(output_aux_gyro_y, target, size_average=False)
        test_loss_aux_gyro_z += model.loss_func(output_aux_gyro_z, target, size_average=False)

        test_loss_aux_max = max(test_loss_aux_acc_x, test_loss_aux_acc_y, test_loss_aux_acc_z, test_loss_aux_gyro_x, test_loss_aux_gyro_y, test_loss_aux_gyro_z)
        test_loss_aux_min = max(test_loss_aux_acc_x, test_loss_aux_acc_y, test_loss_aux_acc_z, test_loss_aux_gyro_x, test_loss_aux_gyro_y, test_loss_aux_gyro_z)

        '''
        if test_loss_main>test_loss_aux_min:
            loss_test+= torch.abs(test_loss_main-test_loss_aux_min) + 5 * test_loss_aux_min
        else:
            loss_test+= test_loss_main + test_loss_aux_acc_x + test_loss_aux_acc_y + test_loss_aux_acc_z + test_loss_aux_gyro_x + test_loss_aux_gyro_y + test_loss_aux_gyro_z
        '''
        loss_test+= test_loss_aux_acc_x + test_loss_aux_acc_y + test_loss_aux_acc_z +test_loss_aux_gyro_x +test_loss_aux_gyro_y +test_loss_aux_gyro_z

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred_aux_acc_x = output_aux_acc_x.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred_aux_acc_y = output_aux_acc_y.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred_aux_acc_z = output_aux_acc_z.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred_aux_gyro_x = output_aux_gyro_x.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred_aux_gyro_y = output_aux_gyro_y.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred_aux_gyro_z = output_aux_gyro_z.data.max(1, keepdim=True)[1] # get the index of the max log-probability

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        correct_aux_acc_x += pred_aux_acc_x.eq(target.data.view_as(pred_aux_acc_x)).cpu().sum()
        correct_aux_acc_y += pred_aux_acc_y.eq(target.data.view_as(pred_aux_acc_y)).cpu().sum()
        correct_aux_acc_z += pred_aux_acc_z.eq(target.data.view_as(pred_aux_acc_z)).cpu().sum()
        correct_aux_gyro_x += pred_aux_gyro_x.eq(target.data.view_as(pred_aux_gyro_x)).cpu().sum()
        correct_aux_gyro_y += pred_aux_gyro_y.eq(target.data.view_as(pred_aux_gyro_y)).cpu().sum()
        correct_aux_gyro_z += pred_aux_gyro_z.eq(target.data.view_as(pred_aux_gyro_z)).cpu().sum()

            #sca_0 = torch.mean(scalar_main[:,0])

 	for i in range(6):
            test_accuracy_list[i] += pred.eq(target.data.view_as(pred))[target == i].cpu().sum().type(torch.FloatTensor)
            test_target_list[i] += (target == i).cpu().sum().type(torch.FloatTensor)

    for i in range(6):
        print('\t target ', i, ' Accuracy = ', test_accuracy_list[i] / test_target_list[i])
            #sca_0 = torch.mean(scalar_main[:,0])

    '''
    for i in range(6):
        fig.add_subplot(1, 6, i + 1)
        plt.hist(scalar_main[:,i].detach(),
            bins=50, 
	    alpha=0.3, 
	    label=str(i))
	    
        plt.legend(loc=1)

        # plt.ylim([0, 200])
	# plt.legend(loc=1)
    plt.draw()
    plt.pause(3)
    plt.cla()
    '''
    test_loss_main /= len(test_loader.dataset)
    test_loss_aux_acc_x /= len(test_loader.dataset)
    test_loss_aux_acc_y /= len(test_loader.dataset)
    test_loss_aux_acc_z /= len(test_loader.dataset)
    test_loss_aux_gyro_x /= len(test_loader.dataset)
    test_loss_aux_gyro_y /= len(test_loader.dataset)
    test_loss_aux_gyro_z /= len(test_loader.dataset)
    #Testing accuracy main model
    correct = correct.numpy()
    correct = correct.astype(float)
    testing_accuracy = round(100.000 *correct/len(test_loader.dataset),4)
    #Testing accuracy aux ACC_X path
    correct_aux_acc_x = correct_aux_acc_x.numpy()
    correct_aux_acc_x = correct_aux_acc_x.astype(float)
    testing_accuracy_aux_acc_x = round(100.000 *correct_aux_acc_x/len(test_loader.dataset),4)
    #Testing accuracy aux ACC_Y path
    correct_aux_acc_y = correct_aux_acc_y.numpy()
    correct_aux_acc_y = correct_aux_acc_y.astype(float)
    testing_accuracy_aux_acc_y = round(100.000 *correct_aux_acc_y/len(test_loader.dataset),4)
    #Testing accuracy aux ACC_Z path
    correct_aux_acc_z = correct_aux_acc_z.numpy()
    correct_aux_acc_z = correct_aux_acc_z.astype(float)
    testing_accuracy_aux_acc_z = round(100.000 *correct_aux_acc_z/len(test_loader.dataset),4)
    #Testing accuracy aux GYRO_X path
    correct_aux_gyro_x = correct_aux_gyro_x.numpy()
    correct_aux_gyro_x = correct_aux_gyro_x.astype(float)
    testing_accuracy_aux_gyro_x = round(100.000 *correct_aux_gyro_x/len(test_loader.dataset),4)
    #Testing accuracy aux GYRO_Y path
    correct_aux_gyro_y = correct_aux_gyro_y.numpy()
    correct_aux_gyro_y = correct_aux_gyro_y.astype(float)
    testing_accuracy_aux_gyro_y = round(100.000 *correct_aux_gyro_y/len(test_loader.dataset),4)
    #Testing accuracy aux GYRO_Z path
    correct_aux_gyro_z = correct_aux_gyro_z.numpy()
    correct_aux_gyro_z = correct_aux_gyro_z.astype(float)
    testing_accuracy_aux_gyro_z = round(100.000 *correct_aux_gyro_z/len(test_loader.dataset),4)

    print('\nTest set: Average total loss: {:.4f}, Accuracy: {}/{} ({}%))\n'.format(
        test_loss_main, correct, len(test_loader.dataset),
        testing_accuracy))
    print('\nTest set: Average acc_x loss: {:.4f}, Accuracy: {}/{} ({}%))\n'.format(
        test_loss_aux_acc_x, correct_aux_acc_x, len(test_loader.dataset),
        testing_accuracy_aux_acc_x))
    print('\nTest set: Average acc_y loss: {:.4f}, Accuracy: {}/{} ({}%))\n'.format(
        test_loss_aux_acc_y, correct_aux_acc_y, len(test_loader.dataset),
        testing_accuracy_aux_acc_y))
    print('\nTest set: Average acc_z loss: {:.4f}, Accuracy: {}/{} ({}%))\n'.format(
        test_loss_aux_acc_z, correct_aux_acc_z, len(test_loader.dataset),
        testing_accuracy_aux_acc_z))
    print('\nTest set: Average gyro_x loss: {:.4f}, Accuracy: {}/{} ({}%))\n'.format(
        test_loss_aux_gyro_x, correct_aux_gyro_x, len(test_loader.dataset),
        testing_accuracy_aux_gyro_x))
    print('\nTest set: Average gyro_y loss: {:.4f}, Accuracy: {}/{} ({}%))\n'.format(
        test_loss_aux_gyro_y, correct_aux_gyro_y, len(test_loader.dataset),
        testing_accuracy_aux_gyro_y))
    print('\nTest set: Average gyro_z loss: {:.4f}, Accuracy: {}/{} ({}%))\n'.format(
        test_loss_aux_gyro_z, correct_aux_gyro_z, len(test_loader.dataset),
        testing_accuracy_aux_gyro_z))

    print('Test main mean scalar is', sca_all)

    return testing_accuracy
def main_dataset():
    train_args = parser_define.train_args_define()
    
    torch.manual_seed(train_args.seed)
    train_args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('current device', train_args.device)
    
    # initial dataset
    har_train_dataset = har_dataset.HAR_Dataset(train=True,noise=True)
    har_trainloader = torch.utils.data.DataLoader(
        har_train_dataset,
        batch_size=train_args.train_batch_size, shuffle=True)
    
    har_test_dataset = har_dataset.HAR_Dataset(train=False, noise=True)
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
