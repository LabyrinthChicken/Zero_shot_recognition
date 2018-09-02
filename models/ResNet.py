from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

__all__ = ['ResNet50','ResNet18']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight, a=0, mode='fan_out')
        nn.init.constant(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal(m.weight, 1.0, 0.02)
            nn.init.constant(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if m.weight is not None:
            init.normal(m.weight.data, std=0.001)
        if m.bias is not None:
            init.constant(m.bias.data, 0.0)

class ResNet50(nn.Module):
    def __init__(self, output_dim, **kwargs):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_in = 2048
        self.num_bottleneck = 512
        self.feat_out = 512

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_in, self.num_bottleneck, bias=True),
            nn.BatchNorm1d(self.num_bottleneck, affine=True),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_bottleneck, output_dim, bias=True)
        )

        self.init()

    def init(self):
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        f = F.avg_pool2d(x, x.size()[2:])
        f = f.squeeze()
        y = self.classifier(f)
        return y

class ResNet18(nn.Module):
    def __init__(self, output_dim, **kwargs):
        super(ResNet18, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=False)
        self.base = nn.Sequential(*list(resnet18.children())[:-2])
        self.feat_in = 512
        self.num_bottleneck = 128
        self.feat_out = 512

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_in, self.num_bottleneck, bias=True),
            nn.BatchNorm1d(self.num_bottleneck, affine=True),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_bottleneck, output_dim, bias=True)
        )

        self.init()

    def init(self):
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        f = F.avg_pool2d(x, x.size()[2:])
        f = f.squeeze()
        y = self.classifier(f)
        return y
