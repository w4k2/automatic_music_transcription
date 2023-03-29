import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnAudio import Spectrogram
from .constants import *
from .Unet_blocks import *
import sys
from .normalization import Normalization
from torchvision.models import resnet18
from .instrument_recognition_model import create_spectrogram_function
from .tensor_visualizer import TensorVisualizer
import random


batchNorm_momentum = 0.1
num_instruments = 1

class ResnetTranscriptionModel(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True, mode='framewise', spec='CQT', norm=1, device='cpu', logdir='./runs'):
        super(ResnetTranscriptionModel, self).__init__()
        self.spectrogram = create_spectrogram_function(spec)
        self.conv = torch.nn.Conv2d(1, 3, (1, 1))
        self.resnet = resnet18(progress=True)
        self.normalize = Normalization(mode)
        self.norm = norm
        self.classification_layer = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 88),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print("INPUT_SHAPE: ", x.shape)
        x = self.conv(input)
        #print("AFTER CONV: ", x.shape)
        x = self.resnet(x)
        feat_resnet = x
        transcription_result = self.classification_layer(feat_resnet)
        feat_resnet = feat_resnet.reshape(32, 1, input.shape[0]//32, 1000)
        #print(f"feat resnet shape: {feat_resnet.shape}")
        return feat_resnet, transcription_result

    def run_on_batch(self, batch, batch_description="training"):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
        if frame_label.dim() == 2:
            frame_label = frame_label.unsqueeze(0)
        spec = self.spectrogram(
            audio_label.reshape(-1, audio_label.shape[-1])[:, :-1])

        spec = torch.log(spec + 1e-5)
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1, -2)
        feat1, pianoroll_reshaped = self(
                    spec.reshape(spec.size(0)*spec.size(1), 1, 1, spec.size(2)))
        pianoroll = pianoroll_reshaped.reshape(spec.size(0), spec.size(1), 88)
        predictions = {
                'onset': pianoroll,
                'frame': pianoroll,
                'feat1': feat1
        }
        #print(f"predictions shape: {predictions['frame'].squeeze(1).shape}, label shape: {frame_label.shape}")
        losses = {
            'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label)
        }
        return predictions, losses, spec

    def freeze_all_layers(self):
        print("Freezing layers is not implemented yet")

    def unfreeze_selected_layers(self, linear=False, lstm=False, conv=False):
        print("Unfreezing layers is not implemented yet")

class UnetTranscriptionModel(nn.Module):
    def __init__(self, ds_ksize, ds_stride, mode='framewise', spec='CQT', norm=1, device='cpu', logdir='./runs', debug_mode = False):
        super(UnetTranscriptionModel, self).__init__()
        global N_BINS
        if spec == 'CQT':
            r = 2
            N_BINS = 88*r
            self.spectrogram = Spectrogram.CQT1992v2(sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                     n_bins=N_BINS, fmin=27.5,
                                                     bins_per_octave=12*r, trainable=False)
        elif spec == 'Mel':
            self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS,
                                                          hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                          trainable_mel=False, trainable_STFT=False)
        else:
            print(f'Please select a correct spectrogram')
        self.tensor_visualizer = TensorVisualizer(logdir=logdir)
        self.global_debug_mode = debug_mode
        self.internal_debug_mode = False
        self.normalize = Normalization(mode)
        self.norm = norm
        self.Unet1_encoder = Encoder(ds_ksize, ds_stride)
        self.Unet1_decoder = Decoder(ds_ksize, ds_stride)
        self.lstm1 = nn.LSTM(
            N_BINS, N_BINS, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(N_BINS*2, 88)

    def forward(self, x):
        x, s, c = self.Unet1_encoder(x)
        feat1 = self.Unet1_decoder(x, s, c)
        x, h = self.lstm1(feat1.squeeze(1))
        head_result = self.linear(x)
        pianoroll = torch.sigmoid(head_result)
        return feat1, pianoroll

    def run_on_batch(self, batch, batch_description="training"):
        global TOTAL_DEBUG_GLOBAL
        audio_label = batch['audio']
        frame_label = batch['frame']

        #in case one dimensional input provided
        if frame_label.dim() == 2:
            frame_label = frame_label.unsqueeze(0)
        if audio_label.dim() == 1:
            audio_label = audio_label.unsqueeze(0)

        spec = self.spectrogram(
            audio_label.reshape(-1, audio_label.shape[-1])[:, :-1])

        # log compression
        spec = torch.log(spec + 1e-5)
        # spectrogram normalization
        spec = self.normalize.transform(spec)
        # swap axes for lstm
        spec = spec.transpose(-1, -2)

        feat1, pianoroll = self(
            spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))

        predictions = {
            'onset': pianoroll,
            'frame': pianoroll,
            'feat1': feat1
        }
        losses = {
            'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label)
        }
        batch_identifier = hash(random.getrandbits(64))
        if self.global_debug_mode and batch_description != "evaluation" and int(batch_description) == 233:
            print("Started to register batches when problem occurs!")
            print(f"first batch: {batch_identifier}")
            self.internal_debug_mode = True
        if self.internal_debug_mode and batch_description != "evaluation":
            copy_of_audio_batch = audio_label.clone().detach()
            copy_of_spec_batch = spec.clone().detach()
            copy_of_frame_batch = frame_label.clone().detach()
            copy_of_path_batch = batch['path']
            copy_of_pianoroll = pianoroll.clone().detach()
            spec_paths = [elem+"_spec.wav" for elem in copy_of_path_batch]
            pred_paths = [elem+"_predictrion.png" for elem in copy_of_path_batch]
            # if epoch_number == "evaluation":
            #     print(f"EVAL SHAPES: audio {copy_of_audio_batch.shape}, spec {copy_of_spec_batch.shape}, label {copy_of_frame_batch.shape}")
            # print(f"TRAIN SHAPES: audio {copy_of_audio_batch.shape}, spec {copy_of_spec_batch.shape}, label {copy_of_frame_batch.shape}")
            for i, spectrogram in enumerate(copy_of_spec_batch):
                self.tensor_visualizer.save_general_audio_from_pytorch_tensor(copy_of_audio_batch[i], copy_of_path_batch[i], batch_identifier, str(batch_description))
                self.tensor_visualizer.save_image_from_pytorch_tensor(spectrogram, spec_paths[i], batch_identifier, str(batch_description))
                self.tensor_visualizer.save_image_from_pytorch_tensor(copy_of_frame_batch[i], copy_of_path_batch[i], batch_identifier, str(batch_description))
                self.tensor_visualizer.save_general_midi_from_pytorch_tensor(copy_of_frame_batch[i], copy_of_path_batch[i], batch_identifier, str(batch_description), "ground_truth_midi")
                self.tensor_visualizer.save_image_from_pytorch_tensor(copy_of_pianoroll[i], pred_paths[i], batch_identifier, str(batch_description))
                self.tensor_visualizer.save_general_midi_from_pytorch_tensor(copy_of_pianoroll[i], pred_paths[i], batch_identifier, str(batch_description), "predicted_midi")

        return predictions, losses, spec

    def freeze_all_layers(self):
        print("Freezing all layers of the model")
        for param in self.parameters():
            param.requires_grad = False
        self.Unet1_encoder.unfreeze_batch_norm()
        self.Unet1_decoder.unfreeze_batch_norm()


    def unfreeze_selected_layers(self, linear=False, lstm=False):
        if linear:
            print("Unfreezing head linear layer of the model!")
            self.head[0].weight.requires_grad = True
            self.head[0].bias.requires_grad = True
            self.head[2].weight.requires_grad = True
            self.head[2].bias.requires_grad = True
            self.head[4].weight.requires_grad = True
            self.head[4].bias.requires_grad = True
            self.head[6].weight.requires_grad = True
            self.head[6].bias.requires_grad = True
        if lstm:
            print("Unfreezing LSTM layer of the model")
            self.lstm1.weight_ih_l0.requires_grad = True
            self.lstm1.weight_hh_l0.requires_grad = True
            self.lstm1.bias_ih_l0.requires_grad = True
            self.lstm1.bias_hh_l0.requires_grad = True
            self.lstm1.weight_ih_l0_reverse.requires_grad = True
            self.lstm1.weight_hh_l0_reverse.requires_grad = True
            self.lstm1.bias_ih_l0_reverse.requires_grad = True
            self.lstm1.bias_hh_l0_reverse.requires_grad = True

    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
            print("Copied ", name, " parameter to target network!")
            i = 0

