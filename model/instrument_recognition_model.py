import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnAudio import Spectrogram
from .constants import *
from .Unet_blocks import *
import sys
import abc
from .normalization import Normalization
from torchvision.models import resnet18

batchNorm_momentum = 0.1
num_instruments = 1


def create_spectrogram_function(spec):
    if spec == 'CQT':
        r = 2
        N_BINS = 88*r
        return Spectrogram.CQT1992v2(sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                     n_bins=N_BINS, fmin=27.5,
                                     bins_per_octave=12*r, trainable=False)
    elif spec == 'Mel':
        return Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS,
                                          hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                          trainable_mel=False, trainable_STFT=False)
    raise Exception("Spectrogram parameter is not correct")


class RecognitionModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'init') and
                callable(subclass.init) and
                hasattr(subclass, 'forward') and
                callable(subclass.forward))


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


class ConvRecognitionModel(nn.Module):
    def init(self, number_of_instruments):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(5, 5), stride=1),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(negative_slope=0.2))
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        # nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(200, 300, kernel_size=(5, 1), stride=2),
            nn.BatchNorm2d(300),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(300, 400, kernel_size=(8, 3), stride=1),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(negative_slope=0.2))
        self.classification_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, number_of_instruments)
        )
        return self

        # nn.ReLU(inplace=True)
        # nn.ReLU(inplace=True),

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pooling(x)
        x = self.conv2(x)
        x = self.max_pooling(x)
        x = self.conv3(x)
        x = self.max_pooling(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classification_layer(x)
        return x


def create_submodel(model_type):
    if model_type == "resnet":
        return ResnetRecognitionModel()
    elif model_type == "conv":
        return ConvRecognitionModel()
    raise Exception(
        f"Recognition model {model_type} is not available!")


class InstrumentRecognitionModel(nn.Module):
    def __init__(self, ds_ksize, ds_stride, mode='framewise', spec='CQT', norm=1, device='cpu', number_of_instruments=10, model_type="resnet"):
        super(InstrumentRecognitionModel, self).__init__()
        self.device = device
        global N_BINS  # using the N_BINS parameter from constant.py

        # Selecting the type of spectrogram to use
        self.spectrogram = create_spectrogram_function(spec)

        self.normalize = Normalization(mode)
        self.norm = norm
        self.loss_function = nn.CrossEntropyLoss()

        self.submodel = create_submodel(model_type)

        self.submodel.init(number_of_instruments)

    def forward(self, x):
        return self.submodel.forward(x)

    def eval(self):
        self.submodel.eval()

    def __is_blacklisted(self, name, blacklist):
        for element in blacklist:
            if element in name:
                return True
        return False

    def load_my_state_dict(self, state_dict, blacklist = []):
        print("Debug - loading state dict to current model!")
        print(f"Parameters not allowed to be transferred: {blacklist}")
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"Warning - {name} not present in model state - skipping!")
                continue
            if self.__is_blacklisted(name, blacklist):
                print(f"Parameter {name} is not allowed to be transfered")
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            print(f"Copying {name} parameter to target network!")
            own_state[name].copy_(param)
            i = 0

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        frame_label = batch['label'].type(torch.LongTensor).to(self.device)
        spec = self.spectrogram(audio_label)
        spec = torch.log(spec + 1e-5)

        spec = self.normalize.transform(spec)

        spec = spec.transpose(-1, -2)
        classification_results = self(
            spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))

        predictions = {
            'results': classification_results
        }
        losses = {
            'loss/transcription': self.loss_function(classification_results, torch.max(frame_label, 1)[1])
        }

        return predictions, losses, spec
