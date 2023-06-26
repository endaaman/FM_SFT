import sys
import re
from typing import NamedTuple, Callable

import torch
import timm
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models


class TimmModel(nn.Module):
    def __init__(self, name, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.name = name
        self.base = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    def get_cam_layers(self):
        if re.match(r'.*efficientnet.*', self.name):
            return [self.base.conv_head]
        if re.match(r'^resnetrs.*', self.name):
            return [self.base.layer4[-1].act3]
        return []

    def forward(self, x, activate=False):
        x = self.base(x)
        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        return x
