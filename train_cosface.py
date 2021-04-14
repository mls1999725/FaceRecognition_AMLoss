from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os

# import utils
import layer
import net
import lfw_eval
from dataset import ImageList

from tensorboardX import SummaryWriter
writer = SummaryWriter()


BatchSize = 512
StepSize = [16000, 24000, 28000]
workers = 4
lr_ori = 0.1
save_path = './checkpoints/'
root_path = ''
train_list = '/home/zwp/MyProject/tf_loss/zwp/result/CASIA-maxpy-clean-112X96.txt'
num_class = 10572  # 10575 - 3


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
    print('Number of Classses: ' + str(num_class))


    # ------------------------------------model--------------------------------------------
    model_ft = net.sphere64a()


    # # --------------load model---------------
    # model_path = './checkpoints/mnface_30_checkpoints.pth'
    # state_dict = torch.load(model_path)
    # model_ft.load_state_dict(state_dict)

    #------------------------------use gpu--------------------
    if use_gpu:
        model_ft = nn.DataParallel(model_ft).cuda()


    # ------------------------------cosface loss and optimizer-------------------------
    MCP = layer.MarginCosineProduct(512, num_class).cuda()
    # MCP = layer.AngleLinear(512, args.num_class).cuda()
    # MCP = torch.nn.Linear(512, args.num_class, bias=False).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([{'params': model_ft.parameters()}, {'params': MCP.parameters()}],
                                lr=lr_ori, momentum=0.9, weight_decay=0.0005)


    for epoch in range(1, 38 + 1):
        # # -------------------my loss----------------------------
        # train(train_loader, model_ft, mining_loss, ce_loss, optimizer, epoch)
        # model_ft.module.save(save_path + 'mnface_' + str(epoch) + '_checkpoints.pth')
        # acc, pred = lfw_eval.eval(save_path + 'mnface_' + str(epoch) + '_checkpoints.pth')

        #-------------------cos face--------------------------
        train(train_loader, model_ft, MCP, criterion, optimizer, epoch)
        model_ft.module.save(save_path + 'cosface_' + str(epoch) + '_checkpoints.pth')
        acc, pred = lfw_eval.eval(save_path + 'cosface_' + str(epoch) + '_checkpoints.pth')


        writer.add_scalar('Test/LFWAcc', acc, epoch)
    print('finished training')


def train(train_loader, model, MCP, criterion, optimizer, epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    loss_display = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        iteration = (epoch - 1) * len(train_loader) + batch_idx

        adjust_learning_rate(optimizer, iteration, StepSize)

        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # compute output
        output = model(data)

        # output = mining_loss(output, target)
        # loss = ce_loss(output, target)

        output = MCP(output, target)
        loss = criterion(output, target)

        loss_display += loss.data
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
            writer.add_scalar('Train/Loss', loss.data, iteration)


def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)

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
