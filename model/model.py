import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from nnAudio import Spectrogram
from .constants import *
from .Unet_blocks import *
import sys
from .normalization import Normalization
from torchvision.models import resnet18
from .instrument_recognition_model import create_spectrogram_function
from .tensor_visualizer import TensorVisualizer
from .decoding import extract_notes_wo_velocity

batchNorm_momentum = 0.1
num_instruments = 1

class ResnetTranscriptionModel(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True, reconstruction=True, mode='framewise', spec='CQT', norm=1, device='cpu',  linear_head=True, conv_head=False, TOTAL_DEBUG=False, logdir="./runs"):
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

    def run_on_batch(self, batch):
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

class NetWithAdditionalHead(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True, reconstruction=True, mode='framewise', spec='CQT', norm=1, device='cpu',  linear_head=True, conv_head=False, TOTAL_DEBUG=False, logdir="./runs"):
        super(NetWithAdditionalHead, self).__init__()
        self.TOTAL_DEBUG = TOTAL_DEBUG
        self.tensor_visualizer = TensorVisualizer(logdir)
        global N_BINS  # using the N_BINS parameter from constant.py

        # Selecting the type of spectrogram to use
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

        self.log = log
        self.normalize = Normalization(mode)
        self.norm = norm
        self.reconstruction = reconstruction
        self.conv_head = conv_head
        self.linear_head = linear_head
        self.WARNING_FLAG = True

        self.Unet1_encoder = Encoder(ds_ksize, ds_stride)
        self.Unet1_decoder = Decoder(ds_ksize, ds_stride)
        self.conv2d_1 = torch.nn.Conv2d(1, N_BINS, (5, 5))
        self.conv2d_2 = torch.nn.Conv2d(N_BINS, N_BINS*2, (3, 3))
        self.transpose_conv1 = torch.nn.ConvTranspose2d(
            N_BINS*2, N_BINS, (3, 3))
        self.transpose_conv2 = torch.nn.ConvTranspose2d(N_BINS, 1, (5, 5))
        self.lstm1 = nn.LSTM(
            N_BINS, N_BINS, batch_first=True, bidirectional=True)

        if reconstruction == True:
            self.Unet2_encoder = Encoder(ds_ksize, ds_stride)
            self.Unet2_decoder = Decoder(ds_ksize, ds_stride)
            self.lstm2 = nn.LSTM(
                88, N_BINS, batch_first=True, bidirectional=True)
            self.linear2 = nn.Linear(N_BINS*2, N_BINS)
        self.linear1 = nn.Linear(N_BINS*2, 88)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(N_BINS*2, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, 88)
        )

    def forward(self, x):
        # U-net 1
        x, s, c = self.Unet1_encoder(x)
        feat1 = self.Unet1_decoder(x, s, c)
        if self.conv_head:
            x = self.conv2d_1(feat1)
            x = self.conv2d_2(x)
            x = self.transpose_conv1(x)
            feat_conv = self.transpose_conv2(x)
            x, h = self.lstm1(feat_conv.squeeze(1))  # remove the channel dim
        else:
            x, h = self.lstm1(feat1.squeeze(1))
        if self.linear_head:
            head_result = self.head(x)
        else:
            head_result = self.linear1(x)
        pianoroll = torch.sigmoid(head_result)  # Use the full LSTM output

        if self.reconstruction:
            # U-net 2
            x, h = self.lstm2(pianoroll)
            # ToDo, remove the sigmoid activation and see if we get a better result
            feat2 = torch.sigmoid(self.linear2(x))
            x, s, c = self.Unet2_encoder(feat2.unsqueeze(1))
            reconstruction = self.Unet2_decoder(x, s, c)  # predict roll

            # Applying U-net 1 to the reconstructed spectrograms
            x, s, c = self.Unet1_encoder(reconstruction)
            feat1b = self.Unet1_decoder(x, s, c)
            x, h = self.lstm1(feat1b.squeeze(1))  # remove the channel dim
            # Use the full LSTM output
            pianoroll2 = torch.sigmoid(self.linear1(x))

            return feat1, feat2, feat1b, reconstruction, pianoroll, pianoroll2
        else:
            if self.conv_head:
                return feat1, feat_conv, pianoroll
            #print(f"BLAX unet feat1 shape {feat1.shape}")
            return feat1, pianoroll

    def run_on_batch(self, batch, epoch_number):
        audio_label = batch['audio']
        frame_label = batch['frame']
        # print(audio_label[0])
        # print(frame_label[0])
        # assert False

        if frame_label.dim() == 2:
            frame_label = frame_label.unsqueeze(0)
        if audio_label.dim() == 1:
            audio_label = audio_label.unsqueeze(0)

        # Converting audio to spectrograms
        # x = torch.rand(8,229, 640)
        spec = self.spectrogram(
            audio_label.reshape(-1, audio_label.shape[-1])[:, :-1])

        # log compression
        if self.log:
            spec = torch.log(spec + 1e-5)

        # Normalizing spectrograms
        spec = self.normalize.transform(spec)

        # swap spec bins with timesteps so that it fits LSTM later
        spec = spec.transpose(-1, -2)  # shape (8,640,229)
        random_number = hash(random.getrandbits(128))
        if(int(random_number) == 1464178761857848575):
            self.WARNING_FLAG = True
        if self.TOTAL_DEBUG and epoch_number != "evaluation" and self.WARNING_FLAG:
            copy_of_audio_batch = audio_label.clone().detach()
            copy_of_spec_batch = spec.clone().detach()
            copy_of_frame_batch = frame_label.clone().detach()
            copy_of_path_batch = batch['path']
            spec_paths = [elem+"_spec.wav" for elem in copy_of_path_batch]
            # if epoch_number == "evaluation":
            #     print(f"EVAL SHAPES: audio {copy_of_audio_batch.shape}, spec {copy_of_spec_batch.shape}, label {copy_of_frame_batch.shape}")
            # print(f"TRAIN SHAPES: audio {copy_of_audio_batch.shape}, spec {copy_of_spec_batch.shape}, label {copy_of_frame_batch.shape}")
            for i, spectrogram in enumerate(copy_of_spec_batch):
                self.tensor_visualizer.save_general_audio_from_pytorch_tensor(copy_of_audio_batch[i], copy_of_path_batch[i], random_number, str(epoch_number))
                self.tensor_visualizer.save_image_from_pytorch_tensor(spectrogram, spec_paths[i], random_number, str(epoch_number))
                self.tensor_visualizer.save_image_from_pytorch_tensor(copy_of_frame_batch[i], copy_of_path_batch[i], random_number, str(epoch_number))
                self.tensor_visualizer.save_general_midi_from_pytorch_tensor(copy_of_frame_batch[i], copy_of_path_batch[i], random_number, str(epoch_number), "ground_truth_midi")

        if self.reconstruction:
            feat1, feat2, feat1b, reconstrut, pianoroll, pianoroll2 = self(
                spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))
            predictions = {
                'onset': pianoroll,
                'frame': pianoroll,
                'frame2': pianoroll2,
                'onset2': pianoroll2,
                'reconstruction': reconstrut,
                'feat1': feat1,
                'feat2': feat2,
                'feat1b': feat1b
            }
            losses = {
                'loss/reconstruction': F.mse_loss(reconstrut.squeeze(1), spec.detach()),
                'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                'loss/transcription2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label)
            }

            return predictions, losses, spec

        else:
            if self.conv_head:
                feat1, feat_conv, pianoroll = self(
                    spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))
            else:
                feat1, pianoroll = self(
                    spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))

            predictions = {
                'onset': pianoroll,
                'frame': pianoroll,
                'feat1': feat1
            }
            if self.TOTAL_DEBUG and epoch_number != "evaluation" and self.WARNING_FLAG:
                copy_of_pianoroll = pianoroll.clone().detach()
                pred_paths = [elem+"_predictrion.png" for elem in copy_of_path_batch]
                for i, path in enumerate(pred_paths):
                    self.tensor_visualizer.save_image_from_pytorch_tensor(copy_of_pianoroll[i], path, random_number, str(epoch_number))
                    self.tensor_visualizer.save_general_midi_from_pytorch_tensor(copy_of_pianoroll[i], path, random_number, str(epoch_number), "predicted_midi")

            if self.conv_head:
                predictions['feat_conv'] = feat_conv
            losses = {
                'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label)
            }
            if self.TOTAL_DEBUG and epoch_number != "evaluation" and self.WARNING_FLAG:
                loss = sum(losses.values())
                if(loss > 0.05):
                    print(f"ALERT FOR {random_number} BATCH!")
            return predictions, losses, spec

    def freeze_all_layers(self):
        print("Freezing all layers of the model")
        for param in self.parameters():
            param.requires_grad = False
        # print("BEFORE")
        # for i, param in enumerate(self.parameters()):
        #     if(param.requires_grad == True):
        #         print(i)
        self.Unet1_encoder.unfreeze_batch_norm()
        self.Unet1_decoder.unfreeze_batch_norm()
        # print("AFTER")
        # for i, param in enumerate(self.parameters()):
        #     if(param.requires_grad == True):
        #         print(i)

    def unfreeze_selected_layers(self, linear=False, lstm=False, conv=False):
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
        if conv:
            print("Unfreezing Conv layers!")
            unfreeze_conv_layer(self.conv2d_1)
            unfreeze_conv_layer(self.conv2d_2)
            unfreeze_conv_layer(self.transpose_conv1)
            unfreeze_conv_layer(self.transpose_conv2)

    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
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


def unfreeze_conv_layer(layer):
    layer.weight.requires_grad = True
    layer.bias.requires_grad = True
