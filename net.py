
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

class Mnist_net(nn.Module):
    def __init__(self):
        super(Mnist_net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*3*3, 2)
        self.ip2 = nn.Linear(2, 10)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*3*3)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        return ip1, F.log_softmax(ip2)

class sphere10a(nn.Module):
    def __init__(self):
        super(sphere10a, self).__init__()

        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        # self.conv1_2 = nn.Conv2d(64,64,3,1,1, bias=False)
        # self.relu1_2 = nn.PReLU(64)
        # self.conv1_3 = nn.Conv2d(64,64,3,1,1, bias=False)
        # self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu2_3 = nn.PReLU(128)

        # self.conv2_4 = nn.Conv2d(128,128,3,1,1, bias=False) #=>B*128*28*24
        # self.relu2_4 = nn.PReLU(128)
        # self.conv2_5 = nn.Conv2d(128,128,3,1,1, bias=False)
        # self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_5 = nn.PReLU(256)

        # self.conv3_6 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*12
        # self.relu3_6 = nn.PReLU(256)
        # self.conv3_7 = nn.Conv2d(256,256,3,1,1, bias=False)
        # self.relu3_7 = nn.PReLU(256)
        #
        # self.conv3_8 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*12
        # self.relu3_8 = nn.PReLU(256)
        # self.conv3_9 = nn.Conv2d(256,256,3,1,1, bias=False)
        # self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        # self.conv4_2 = nn.Conv2d(512,512,3,1,1, bias=False)
        # self.relu4_2 = nn.PReLU(512)
        # self.conv4_3 = nn.Conv2d(512,512,3,1,1, bias=False)
        # self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6, 2)

        # weight initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        # x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        # x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        # x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        # x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        # x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class sphere20a(nn.Module):
    def __init__(self):
        super(sphere20a, self).__init__()

        #input = B*3*112*112
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*56
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1, bias=False)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1, bias=False)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*28
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1, bias=False) #=>B*128*28*28
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*14
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*14
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*14
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*14
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*7
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1, bias=False)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1, bias=False)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*7,512)

        # weight initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        # print(x.size())
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        # print(x.size())
        x = self.fc5(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

class sphere64a(nn.Module):
    def __init__(self):
        super(sphere64a, self).__init__()

        # input = B * 3 * 112 * 96

        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1, bias=False)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1, bias=False)
        self.relu1_3 = nn.PReLU(64)

        self.conv1_4 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu1_4 = nn.PReLU(64)
        self.conv1_5 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu1_5 = nn.PReLU(64)

        self.conv1_6 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu1_6 = nn.PReLU(64)
        self.conv1_7 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu1_7 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1, bias=False) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu2_5 = nn.PReLU(128)

        self.conv2_6 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_6 = nn.PReLU(128)
        self.conv2_7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_7 = nn.PReLU(128)

        self.conv2_8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_8 = nn.PReLU(128)
        self.conv2_9 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_9 = nn.PReLU(128)

        self.conv2_10 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_10 = nn.PReLU(128)
        self.conv2_11 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_11 = nn.PReLU(128)

        self.conv2_12 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_12 = nn.PReLU(128)
        self.conv2_13 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_13 = nn.PReLU(128)

        self.conv2_14 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_14 = nn.PReLU(128)
        self.conv2_15 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_15 = nn.PReLU(128)

        self.conv2_16 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_16 = nn.PReLU(128)
        self.conv2_17 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_17 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1, bias=False) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1, bias=False)
        self.relu3_9 = nn.PReLU(256)

        self.conv3_10 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_10 = nn.PReLU(256)
        self.conv3_11 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_11 = nn.PReLU(256)

        self.conv3_12 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_12 = nn.PReLU(256)
        self.conv3_13 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_13 = nn.PReLU(256)

        self.conv3_14 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_14 = nn.PReLU(256)
        self.conv3_15 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_15 = nn.PReLU(256)

        self.conv3_16 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_16 = nn.PReLU(256)
        self.conv3_17 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_17 = nn.PReLU(256)

        self.conv3_18 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_18 = nn.PReLU(256)
        self.conv3_19 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_19 = nn.PReLU(256)

        self.conv3_20 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_20 = nn.PReLU(256)
        self.conv3_21 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_21 = nn.PReLU(256)

        self.conv3_22 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_22 = nn.PReLU(256)
        self.conv3_23 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_23 = nn.PReLU(256)

        self.conv3_24 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_24 = nn.PReLU(256)
        self.conv3_25 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_25 = nn.PReLU(256)

        self.conv3_26 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_26 = nn.PReLU(256)
        self.conv3_27 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_27 = nn.PReLU(256)

        self.conv3_28 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_28 = nn.PReLU(256)
        self.conv3_29 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_29 = nn.PReLU(256)

        self.conv3_30 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_30 = nn.PReLU(256)
        self.conv3_31 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_31 = nn.PReLU(256)

        self.conv3_32 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_32 = nn.PReLU(256)
        self.conv3_33 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_33 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1, bias=False)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1, bias=False)
        self.relu4_3 = nn.PReLU(512)

        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_4 = nn.PReLU(512)
        self.conv4_5 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_5 = nn.PReLU(512)

        self.conv4_6 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_6 = nn.PReLU(512)
        self.conv4_7 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_7 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6, 512)

        # self.fc6 = nn.Linear(512, self.classnum, bias=False)

        # weight initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    m.weight.data.normal_(0, 0.01)



    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        x = x + self.relu1_5(self.conv1_5(self.relu1_4(self.conv1_4(x))))
        x = x + self.relu1_7(self.conv1_7(self.relu1_6(self.conv1_6(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
        x = x + self.relu2_7(self.conv2_7(self.relu2_6(self.conv2_6(x))))
        x = x + self.relu2_9(self.conv2_9(self.relu2_8(self.conv2_8(x))))
        x = x + self.relu2_11(self.conv2_11(self.relu2_10(self.conv2_10(x))))
        x = x + self.relu2_13(self.conv2_13(self.relu2_12(self.conv2_12(x))))
        x = x + self.relu2_15(self.conv2_15(self.relu2_14(self.conv2_14(x))))
        x = x + self.relu2_17(self.conv2_17(self.relu2_16(self.conv2_16(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))
        x = x + self.relu3_11(self.conv3_11(self.relu3_10(self.conv3_10(x))))
        x = x + self.relu3_13(self.conv3_13(self.relu3_12(self.conv3_12(x))))
        x = x + self.relu3_15(self.conv3_15(self.relu3_14(self.conv3_14(x))))
        x = x + self.relu3_17(self.conv3_17(self.relu3_16(self.conv3_16(x))))
        x = x + self.relu3_19(self.conv3_19(self.relu3_18(self.conv3_18(x))))
        x = x + self.relu3_21(self.conv3_21(self.relu3_20(self.conv3_20(x))))
        x = x + self.relu3_23(self.conv3_23(self.relu3_22(self.conv3_22(x))))
        x = x + self.relu3_25(self.conv3_25(self.relu3_24(self.conv3_24(x))))
        x = x + self.relu3_27(self.conv3_27(self.relu3_26(self.conv3_26(x))))
        x = x + self.relu3_29(self.conv3_29(self.relu3_28(self.conv3_28(x))))
        x = x + self.relu3_31(self.conv3_31(self.relu3_30(self.conv3_30(x))))
        x = x + self.relu3_33(self.conv3_33(self.relu3_32(self.conv3_32(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = x + self.relu4_5(self.conv4_5(self.relu4_4(self.conv4_4(x))))
        x = x + self.relu4_7(self.conv4_7(self.relu4_6(self.conv4_6(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)

        # feature = x
        # x = self.fc6(x)
        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


