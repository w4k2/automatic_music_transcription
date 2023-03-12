import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import abc
from nnAudio import Spectrogram
from .constants import *
from torchvision.models import resnet18


class RecognitionModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'init') and
                callable(subclass.init) and
                hasattr(subclass, 'forward') and
                callable(subclass.forward) and
                hasattr(subclass, 'load_my_state_dict') and
                callable(subclass.load_my_state_dict))


class ResnetRecognitionModel(nn.Module):
    def init(self, number_of_instruments):
        self.conv = torch.nn.Conv2d(1, 3, (1, 1))
        self.resnet = resnet18(progress=True)
        self.classification_layer = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, number_of_instruments)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        x = self.classification_layer(x)
        return x

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            print("Copied ", name, " parameter to target network!")
            i = 0
