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
    print ("Training... Epoch = %d" % epoch)
    ip1_loader = []
    idx_loader = []
    acc = 0.0

    for i, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        ip1, pred = model(data)
        loss = criterion[0](pred, target) + criterion[1](target, ip1)

        _, p = torch.max(pred, 1)
        bacc = torch.sum(p == target)
        acc += bacc.data[0]

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        loss.backward()

        optimizer[0].step()
        optimizer[1].step()

        ip1_loader.append(ip1)
        idx_loader.append((target))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    # print "   acc = %.3f" % acc
    # print "   len=%d" % len(labels)
    acc = float(acc) / len(labels)
    print "   train_acc = %.3f" % acc*100
    visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch, acc)

def test(test_loader, model, use_cuda, lentestset):
    acc = 0.0
    centerf = torch.zeros(10, 2)
    for i, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
            centerf = centerf.cuda()
        data, target = Variable(data), Variable(target)

        ip1, pred = model(data)
        for j in range(len(target)):
            centerf[target[j]]+=ip1[j]

        _, p = torch.max(pred, 1)
        bacc = torch.sum(p == target)
        acc += bacc.data[0]
    acc = float(acc) / lentestset
    print "   test_acc = %.3f" % acc*100

    intrad = 0.0
    interd = 0.0

    for i, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        ip1, pred = model(data)
        for j in range(len(target)):
            intrad += torch.norm(ip1[j]-centerf[target[j]], 2)
            interd += torch.min(torch.norm(ip1[j]-centerf, p=2, dim=1))

        _, p = torch.max(pred, 1)
        bacc = torch.sum(p == target)
        acc += bacc.data[0]
    intrad = float(intrad) / lentestset
    interd = float(interd) / lentestset
    print("  intrad = %.3f" % intrad)
    print("  interd = %.3f" % interd)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # use_gpu = torch.cuda.is_available()

    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    # Dataset
    trainset = datasets.MNIST('../../data', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    testset = datasets.MNIST('../../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=4)

    # Model
    model = net.Mnist_net()

    # NLLLoss
    nllloss = nn.NLLLoss()  # CrossEntropyLoss = log_softmax + NLLLoss
    # CenterLoss
    loss_weight = 1.0
    centerloss = CenterLoss(10, 2, loss_weight)
    if use_cuda:
        nllloss = nllloss.cuda()
        centerloss = centerloss.cuda()
        model = model.cuda()
    criterion = [nllloss, centerloss]

    # optimzer4nn
    optimizer4nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    sheduler = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.8)

    # optimzer4center
    optimzer4center = optim.SGD(centerloss.parameters(), lr=0.5)

    fig = plt.figure()
    for epoch in range(50):
        sheduler.step()
        # print optimizer4nn.param_groups[0]['lr']
        train(train_loader, model, criterion, [optimizer4nn, optimzer4center], epoch + 1, use_cuda)
        test(test_loader, model, use_cuda, len(testset))

if __name__ == '__main__':
    main()
