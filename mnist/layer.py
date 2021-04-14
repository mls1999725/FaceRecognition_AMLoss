import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.autograd.function import Function

class miniloss(nn.Module):
    """
    Args:
        in_features : 512
        out_features : 10558
        s : norm of input feature

    """

    def __init__(self, in_features, out_features, s=64, m1=0.5, m2=0.5):
        super(miniloss, self).__init__()
        self.in_features = in_features  # 2
        self.out_fearures = out_features # 10
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)).cuda()
        # nn.init.kaiming_normal_(self.weight)
        # self.weight = nn.Parameter(torch.randn(out_features, in_features)).cuda()
        nn.init.xavier_uniform_(self.weight)
        self.m1 = m1
        self.m2 = m2
        # self.weight.data.normal_(0, 0.01)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input, dim=1, eps=1e-12), F.normalize(self.weight, dim=1, eps=1e-12))
        # alpha = -0.3 * torch.pow(cosine, 2) + 0.3
        # m1 = 0.5
        # m2 = 0.9
        theta = torch.acos(cosine)

        alpha = self.m1/2 * torch.cos(2 * theta + math.pi) + self.m1/2
        beta = self.m2/2 * torch.cos(2 * theta) + self.m2/2
        phi = cosine - alpha
        phj = cosine + beta
        # -------------------convert label to onehot-----------------
        one_hot = Variable(torch.zeros(cosine.size()))
        one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * phj)
        output *= self.s

        # output = F.linear(input, self.weight)
        return output

        # return output, torch.sum(one_hot * theta, 1).cpu().detach().numpy(), torch.max((1.0 - one_hot) * theta, 1)[
        #     0].cpu().detach().numpy()


        # gassian
        # alpha = m1 * torch.exp(-torch.pow(theta-60*math.pi/180, 2)/(2*pow(15*math.pi/180, 2)))
        # beta = m2 * torch.exp(-torch.pow(theta - 110 * math.pi / 180, 2) / (2 * pow(10 * math.pi / 180, 2)))

        # # arcface
        # alpha = 0.17
        # beta = 0.0
        # phi = torch.cos(theta + alpha)
        # phj = torch.cos(theta - beta)
        # #-------------------convert label to onehot-----------------
        # one_hot = Variable(torch.zeros(cosine.size()))
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        # one_hot.scatter_(1, label.view(-1, 1), 1)
        # output = (one_hot * phi) + ((1.0 - one_hot) * phj)
        # output *= self.s
        #
        # return output, torch.sum(one_hot * theta, 1).cpu().detach().numpy(), torch.max((1.0-one_hot) * theta, 1)[0].cpu().detach().numpy()


class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.30): # s=30.0  m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # cosine = F.linear(input, self.weight)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = Variable(torch.zeros(cosine.size()))
        one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = Variable(torch.zeros(cos_theta.size()))
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        loss = self.centerlossfunc(feat, label, self.centers)
        loss /= (batch_size if self.size_average else 1)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new(centers.size(0)).fill_(1)
        ones = centers.new(label.size(0)).fill_(1)
        grad_centers = centers.new(centers.size()).fill_(0)

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff, None, grad_centers