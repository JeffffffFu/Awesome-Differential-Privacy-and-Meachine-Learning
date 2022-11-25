#!/user/bin/python
# author jeff

from torch import nn
import torch.nn.functional as F
import torch
#两个卷积、两个池化、两个全连接层的卷积神经网络
##1、init函数中要说明输入和输出in_ch out_ch
##2、在forward函数中把各个部分连接起来
##注意如果要用这个model,MNIST和fashionMnist数据的维度要进行reshap为（x,1,28,28）
# #如果要进行CNN神经网络，输入要进行reshap满足这个网络的输入要求,下面这段代码可在dataset.py里
# x_train=x_train.reshape(50000,1,28,28)
# x_valid=x_valid.reshape(10000,1,28,28)
# x_test=x_test.reshape(10000,1,28,28)
#这个针对Mnist和fashionMnist,因为他们是灰色的，没有彩色通道

#根据《Tempered Sigmoid Activations for Deep Learning with Differential Privacy》写的CNN，用于Mnist和fashionMnist数据集
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        x=self.conv(x)
        return x

#基于文章《Tempered Sigmoid Activations for Deep Learning with Differential Privacy》将激活函数改为Tanh
class CNN_tanh(nn.Module):
    def __init__(self):
        super(CNN_tanh, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.Tanh(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.Tanh(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        x=self.conv(x)
        return x


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x




#根据《Tempered Sigmoid Activations for Deep Learning with Differential Privacy》写的CNN，用于CIFAR10数据集
class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128*4*4, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 10, bias=True), )

    def forward(self,x):
        x=self.conv(x)
        return x


#根据《Tempered Sigmoid Activations for Deep Learning with Differential Privacy》写的CNN，用于CIFAR10数据集，激活函数改为tanh
class Cifar10CNN_tanh(nn.Module):
    def __init__(self):
        super(Cifar10CNN_tanh, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128*4*4, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 10, bias=True), )

    def forward(self,x):
        x=self.conv(x)
        return x


#文章《Large Language Models Can Be Strong Differentially Private Learners》提供的用于Mnist和fashionMnist的模型
#调用方式    centralized_model = CIFAR10_CNN(1, input_norm=None, num_groups=None, size=None)
class MNIST_CNN(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)

                layers += [conv2d, nn.Tanh()]
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        nn.Tanh(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


#文章《Large Language Models Can Be Strong Differentially Private Learners》提供的用于CIFAR10的模型
#调用方式    centralized_model = CIFAR10_CNN(3, input_norm=None, num_groups=None, size=None)
#跑CIFAR10用这个模型效果比较好，跑MNIST和FASHIONMnist用上面自己的模型即可
class CIFAR10_CNN(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x