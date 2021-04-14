import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from CenterLoss import CenterLoss
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import net, os
import layer
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
feature_dim = 512 #512
epo = 40

def visualize(feat, labels, epoch, acc):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    # fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(10):
        ax.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    # ax.legend(bbox_to_anchor=(1.0, 0.8), loc='upper left')
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], bbox_to_anchor=(1.0, 0.8), loc='upper left')
    # plt.xlim(xmin=-5, xmax=5)
    # plt.ylim(ymin=-5, ymax=5)
    # ax.text(-6.8, 6.1, "epoch=%d" % epoch)
    # ax.text(-6.8, 6.1, "acc=%.3f" % acc)
    plt.savefig('./images/epoch=%d.png' % epoch)
    plt.draw()
    plt.pause(0.001)
    # plt.close()


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    #print "Training... Epoch = %d" % epoch
    ip1_loader = []
    idx_loader = []
    acc = 0.0
    loss_display = 0.0

    for i, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        ip1 = model(data)
        output = criterion[0](ip1, target)
        # output, theta, beta = criterion[0](ip1, target)
        loss = criterion[1](output, target)

        _, p = torch.max(F.log_softmax(output, dim=1), 1)
        bacc = torch.sum(p == target)
        acc += bacc.data
        loss_display += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ip1_loader.append(ip1)
        idx_loader.append((target))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    # print "   acc = %.3f" % acc
    # print "   len=%d" % len(labels)
    acc = float(acc) / len(labels)
    loss_display = float(loss_display) / len(labels)
    if epoch == epo:
        print "   train_acc = %.3f  train_loss = %.3f" % (acc*100, loss_display)
    # visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch, acc)

def test(test_loader, model, criterion, use_cuda, lentestset, epoch):
    acc = 0.0
    centerf = torch.zeros(10, feature_dim)
    cnt = torch.zeros(10, 1)
    loss_display = 0.0
    # print(test_loader[1])
    for i, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
            centerf = centerf.cuda()
            cnt = cnt.cuda()
        data, target = Variable(data), Variable(target)

        ip1 = F.normalize(model(data), dim=1)

        # output, theta, beta = criterion[0](ip1, target)
        output = criterion[0](ip1, target)
        loss = criterion[1](output, target)
        loss_display += loss.data
        _, p = torch.max(F.log_softmax(output, dim=1), 1)

        for j in range(len(target)):
            centerf[target[j]]+=ip1[j]
            cnt[target[j]] += 1

        bacc = torch.sum(p == target)
        acc += bacc.data

    acc = float(acc) / lentestset
    loss_display /= lentestset
    if epoch == epo:
        print "   test_acc = %.3f  test_loss = %.3f" % (acc*100, loss_display)

    for i in range(10):
        centerf[i] = torch.div(centerf[i], cnt[i])

    intrad = 0.0
    interd = 0.0

    for i, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        # ip1 = model(data)

        ip1 = F.normalize(model(data), dim=1)
        for j in range(len(target)):
            # tmp = target[j]
            dist = torch.norm(ip1[j]-centerf, p=2, dim=1)
            intrad += dist[target[j]]

            # intrad += torch.norm(ip1[j] - centerf[target[j]], 2).cpu()
            # print(torch.norm(ip1[j] - centerf[target[j]], 2).cpu())
            # dist = torch.norm(ip1[j]-centerf, p=2, dim=1)
            # print dist
            # print target[j]
            dist[target[j]] = torch.max(dist)
            # print(torch.min(dist).cpu())
            interd += torch.min(dist).cpu()


            # interd += torch.min(torch.norm(ip1[j]-centerf, p=2, dim=1))

    intrad = float(intrad) / lentestset
    interd = float(interd) / lentestset
    if epoch == epo:
        print"  intrad = %.3f   interd = %.3f" % (intrad, interd)
    return intrad, interd


def main():

    # use_gpu = torch.cuda.is_available()

    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    # Dataset
    trainset = datasets.MNIST('./data', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    testset = datasets.MNIST('./data', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)# 16
    test_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=4)

    tosave_intra = np.zeros((9, 9))
    tosave_inter = np.zeros((9, 9))

    CIR = 5
    for x in range(1, 10, 1):
        for y in range(1, 10, 1):
            intrad_tmp = 0.0
            interd_tmp = 0.0
            m1 = 0.1 * x
            m2 = 0.1 * y
            print "m1 = %.2f   m2 = %.2f" % (m1, m2)
            for cicle in range(1, 1+CIR, 1):
               # Model
               model = net.Mnist_net2()
               # model = net.sphere10a()

               mining_loss = layer.miniloss(feature_dim, 10, m1=m1, m2=m2)
               ce_loss = nn.CrossEntropyLoss()
               if use_cuda:
                  mining_loss = mining_loss.cuda()
                  ce_loss = ce_loss.cuda()
                  model = model.cuda()
               criterion = [mining_loss, ce_loss]
               optimizer = optim.SGD([{'params': model.parameters()}, {'params': mining_loss.parameters()}],
                          lr=0.01, momentum=0.5)
               # optimizer4nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
               sheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.8)

               # fig = plt.figure()

               for epoch in range(1, epo+1):
                  sheduler.step()
                  # print optimizer4nn.param_groups[0]['lr']
                  train(train_loader, model, criterion, optimizer, epoch, use_cuda)
               intrad, interd = test(test_loader, model, criterion, use_cuda, len(testset), epoch)
               intrad_tmp+=intrad
               interd_tmp+=interd
            intrad_tmp/=CIR
            interd_tmp/=CIR
            tosave_intra[x-1][y-1]=intrad_tmp
            tosave_inter[x-1][y-1]=interd_tmp
    np.savetxt('./intrad.csv',tosave_intra, delimiter=',')
    np.savetxt('./interd.csv', tosave_inter, delimiter=',')


if __name__ == '__main__':
    main()
