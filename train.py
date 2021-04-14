from __future__ import print_function, division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,7'
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time

# import utils
import layer
import net
import lfw_eval
from dataset import ImageList

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import numpy as np
import pickle

from tensorboardX import SummaryWriter
writer = SummaryWriter()


BatchSize = 512
StepSize = [15000, 24000]
#StepSize = [10000, 20000]
workers = 4
lr_ori = 0.1
save_path = './checkpoints/'
root_path = ''
train_list = '/media/data4/zhangzhenduo/dataset/traindataset/CASIA_aligned.csv'
num_class = 10575  # 10575 - 3
multi_sphere = True
f = open("./Train_CASIA_log_m10.3_m20.0_multi_debug.txt", 'w+')

use_gpu = torch.cuda.is_available()

def main():
    # ----------------------------------------load images----------------------------------------

    train_loader = torch.utils.data.DataLoader(
        ImageList(root=root_path, fileList=train_list,
                  transform=transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                  ])),
        batch_size=BatchSize, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)

    print('length of train Dataset: ' + str(len(train_loader.dataset)))
    f.write('length of train Dataset: ' + str(len(train_loader.dataset)) + '\n')
    print('Number of Classses: ' + str(num_class))
    f.write('Number of Classses: ' + str(num_class) + '\n')


    # ------------------------------------model--------------------------------------------
    model_ft = net.sphere20a()


    # # --------------load model---------------
    # model_path = './checkpoints/mnface_30_checkpoints.pth'
    # state_dict = torch.load(model_path)
    # model_ft.load_state_dict(state_dict)

    #------------------------------use gpu--------------------
    if use_gpu:
        # speed up training
        model_ft = nn.DataParallel(model_ft).cuda()
        # model_ft = model_ft.cuda()


    # -----------------------------------loss function and optimizer--------------------------

    if multi_sphere:
        mining_loss = layer.MultiMini(512, num_class)
    else:
        mining_loss = layer.miniloss(512, num_class)
    ce_loss = nn.CrossEntropyLoss()
    if use_gpu:
        mining_loss = mining_loss.cuda()
        ce_loss = ce_loss.cuda()

    optimizer = optim.SGD([{'params': model_ft.parameters()}, {'params': mining_loss.parameters()}],
                          lr=lr_ori, momentum=0.9, weight_decay=0.0005)


    # # ------------------------------cosface loss and optimizer-------------------------
    # MCP = layer.MarginCosineProduct(512, num_class).cuda()
    # # MCP = layer.AngleLinear(512, args.num_class).cuda()
    # # MCP = torch.nn.Linear(512, args.num_class, bias=False).cuda()
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD([{'params': model_ft.parameters()}, {'params': MCP.parameters()}],
    #                             lr=lr_ori, momentum=0.9, weight_decay=0.0005)


    for epoch in range(1, 30 + 1):
        # -------------------my loss----------------------------
        # x, y = train(train_loader, model_ft, mining_loss, ce_loss, optimizer, epoch)
        # if multi-sphere
        train(train_loader, model_ft, mining_loss, ce_loss, optimizer, epoch)

        model_ft.module.save(save_path + 'mnface_' + str(epoch) + '_checkpoints.pth')
        acc = lfw_eval.eval(model_path = save_path + 'mnface_' + str(epoch) + '_checkpoints.pth')

        # if epoch in [1,2,3,10,15,20,30]:
            # pickle.dump(x, open("/home/baixy/Codes/class-invariant-loss/xarc"+str(epoch)+".pkl", 'wb'))
            # pickle.dump(y, open("/home/baixy/Codes/class-invariant-loss//yarc" + str(epoch) + ".pkl", 'wb'))
        # del x

        # #-------------------cos face--------------------------
        # train(train_loader, model_ft, MCP, criterion, optimizer, epoch)
        # model_ft.module.save(save_path + 'cosface_' + str(epoch) + '_checkpoints.pth')
        # acc, pred = lfw_eval.eval(save_path + 'cosface_' + str(epoch) + '_checkpoints.pth')


        writer.add_scalar('Test/LFWAcc', acc, epoch)

    # fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    # ax0.hist(x1, 100, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
    # ax1.hist(x10, 100, normed=1, histtype='bar', facecolor='pink', alpha=0.75)
    # ax2.hist(x20, 100, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
    # plt.show()
    print('finished training')
    f.write("finished training" + '\n')
    f.close()


def train(train_loader, model, mining_loss, ce_loss, optimizer, epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    loss_display = 0.0
    # x = np.empty([0,])
    # y = np.empty([0,])

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        iteration = (epoch - 1) * len(train_loader) + batch_idx

        adjust_learning_rate(optimizer, iteration, StepSize)

        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # compute output
        output = model(data)
        # print(output.size())
        # print(target)
        # output, theta , beta = mining_loss(output, target)
        
        outputs = mining_loss(output, target)
        # print(len(outputs))

        # if epoch == 1 or epoch == 2 or epoch == 3 or epoch==15 or epoch==30:
        # if epoch in [1, 2, 3, 10, 15, 20, 30]:
            # x = np.concatenate((x, theta), axis=0)
            # y = np.concatenate((y, beta), axis=0)

        loss = ce_loss(outputs[0], target)
        # if multi-sphere
        for output in outputs[1:]:
            loss = loss + ce_loss(output, target)

        # output = MCP(output, target)
        # loss = criterion(output, target)

        loss_display += loss.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            time_used = time.time() - time_curr
            loss_display /= 100
            # INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(MCP.m, MCP.s)
            # INFO = ' lambda: {:.4f}'.format(MCP.lamb)
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_display, time_used, 100) #+ INFO
            )
            time_curr = time.time()
            loss_display = 0.0
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), iteration)
    # return x, y


def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)
    f.write(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string + '\n')

def adjust_learning_rate(optimizer, iteration, step_size):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = lr_ori * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass

if __name__ == '__main__':
    main()
