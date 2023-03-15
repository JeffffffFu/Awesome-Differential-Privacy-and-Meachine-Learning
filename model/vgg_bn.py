"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''


# import torch
# import torch.nn as nn
# from preact_resnet import PreActResNet18
# from vgg_bn import vgg19_bn
# from typing import Any, Callable, List, Optional, Union, Tuple
# from functools import partial
# import PIL.Image as Image
# import os
# import pandas


import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim

# from dataloader import UTKFaceDataset

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features,  num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        # print(x.shape)
        output = self.features(x)
        # print(output.shape)
        output = output.view(output.size()[0], -1)
        # print(output.shape)
        output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False, input_channel=3):
    layers = []

    # input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['A'], batch_norm=True, input_channel=input_channel), num_class=num_classes)


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19_bn(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['E'], batch_norm=True, input_channel=input_channel), num_class=num_classes)


def get_utkface():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = UTKFaceDataset(root="../data", attr="race", transform=transform)
    input_channel = 3
    num_classes = 4
    return num_classes, dataset, dataset, input_channel


def train_vgg():
    # num_classes = 10
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # # dataset = train_set + test_set
    # input_channel = 3
    # train_set = torchvision.datasets.CIFAR10(
    #     root="./data", train=True, download=True, transform=transform)
    # test_set = torchvision.datasets.CIFAR10(
    #     root="./data", train=False, download=True, transform=transform)

    # num_classes = 10
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    # train_set = torchvision.datasets.FashionMNIST(
    #     root="./data", train=True, download=True, transform=transform)
    # test_set = torchvision.datasets.FashionMNIST(
    #     root="./data", train=False, download=True, transform=transform)

    # # dataset = train_set + test_set
    # input_channel = 1

    num_classes, train_set, test_set, input_channel = get_utkface()

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg19_bn(input_channel, num_classes).to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    epochs = 30

    train_losses = []
    test_losses = []
    best_accuracy = 0
    # 训练测试
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            pre = model(images)
            loss = criterion(pre, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if e % 3 == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)  # 最高的 概率，对应的类别
                    equals = top_class == labels.view(top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            accuracy = accuracy / len(test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(),
                           "./data/model/vgg/model_state_dict.pth")
            print("Epoch: {}/{}..".format(e + 1, epochs))
            print("Training Loss: {:.3f}".format(
                running_loss / len(train_loader)))
            print("Test Loss: {:.3f}".format(test_loss / len(test_loader)))
            print("Accuracy: {:.3f}".format(accuracy))
            print("-----------------------------------------------------")


if __name__ == "__main__":
    train_vgg()
