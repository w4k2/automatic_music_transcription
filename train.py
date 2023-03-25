import copy
import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fail_observer import FailObserver
from model import *
from model.dataset import (OriginalMAPS, SynthesizedInstruments,
                           SynthesizedTrumpet)
from model.evaluate_fn import evaluate_wo_velocity
from snapshot import Snapshot

ex = Experiment('train_transcriber')


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# parameters for the network
ds_ksize, ds_stride = (2, 2), (2, 2)
mode = 'imagewise'
sparsity = 1


@ex.config
def config():
    # Choosing GPU to use
    GPU = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    device = 'cuda'
    dataset_root_dir = "."
    spec = 'CQT'
    resume_iteration = None
    train_on = 'GuitarSet'
    pretrained_model_path = None
    freeze_all_layers = False
    unfreeze_linear = False
    unfreeze_lstm = False
    unfreeze_conv = False
    conv_head = False
    linear_head = True
    TOTAL_DEBUG = False

    batch_size = 32
    sequence_length = 327680
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(
            f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    epoches = 2000
    learning_rate = 0.01
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98
    model_type = "unet"
    reconstruction = False

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    refresh = False
    # set random seed for whole experiment
    seed = 33
    seed_everything(seed)

    destination_dir = "runs"
    logdir = f'{destination_dir}/TRAIN_TRANSCRIPTION_{model_type}_ON_{train_on}_{spec}_{mode}_' + \
        datetime.now().strftime('%y%m%d-%H%M%S')
    fail_observer = FailObserver(logdir=logdir, snapshot_capacity=10, TOTAL_DEBUG=TOTAL_DEBUG)
    ex.observers.append(FileStorageObserver.create(logdir))
    ex.observers.append(fail_observer)


def detect_epoch(filename):
    only_model_name = filename.split("/")[-1]
    return only_model_name[6:-3]


def create_transcription_datasets(dataset_type):
    if dataset_type == "MAESTRO":
        return [(MAESTRO, ['train']), (MAPS, ['ENSTDkAm', 'ENSTDkCl']), (MAPS, ['ENSTDkAm', 'ENSTDkCl'])]
    elif dataset_type == "MusicNet":
        return [(MusicNet, ['train']), (MusicNet, ['test']), (MAPS, ['ENSTDkAm', 'ENSTDkCl'])]
    elif dataset_type == "GuitarSet":
        return [(GuitarSet, ['train']), (GuitarSet, ['val']), (GuitarSet, ['test'])]
    elif dataset_type == "SynthesizedTrumpet":
        return [(SynthesizedTrumpet, ['train']), (SynthesizedTrumpet, ['val']), (SynthesizedTrumpet, ['test'])]
    elif dataset_type == "SynthesizedInstruments":
        return [(SynthesizedInstruments, ['train']), (SynthesizedInstruments, ['val']), (SynthesizedInstruments, ['test'])]
    elif dataset_type == "MAPS":
        return [(MAPS, ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']),
                (MAPS, ['ENSTDkAm', 'ENSTDkCl']),
                (MAPS, ['ENSTDkAm', 'ENSTDkCl'])]
    elif dataset_type == "OriginalMAPS":
        return [(OriginalMAPS, ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']),
                (OriginalMAPS, ['ENSTDkAm', 'ENSTDkCl']),
                (OriginalMAPS, ['ENSTDkAm', 'ENSTDkCl'])]


def create_model(model_type):
    if model_type == "resnet":
        print("Using resnet transcription model")
        return ResnetTranscriptionModel
    else:  # fallback for unet
        print("Using unet transcription model")
        return NetWithAdditionalHead


@ex.automain
def train(spec, resume_iteration, train_on, pretrained_model_path, freeze_all_layers, unfreeze_linear, unfreeze_lstm,
          unfreeze_conv, batch_size, sequence_length, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate,
          leave_one_out, clip_gradient_norm, validation_length, refresh, device, reconstruction, epoches, logdir, linear_head, conv_head,
          dataset_root_dir, model_type, fail_observer, TOTAL_DEBUG):
    print_config(ex.current_run)
    print("Reconstruction: ", reconstruction)

    dataset_data = create_transcription_datasets(dataset_type=train_on)
    TrainDataset = dataset_data[0][0]
    train_dataset_groups = dataset_data[0][1]
    ValidationDataset = dataset_data[1][0]
    val_dataset_groups = dataset_data[1][1]
    TestDataset = dataset_data[2][0]
    test_dataset_groups = dataset_data[2][1]

    train_dataset = TrainDataset(dataset_root_dir=dataset_root_dir, groups=train_dataset_groups,
                                 sequence_length=sequence_length, device=device, refresh=refresh, TOTAL_DEBUG=TOTAL_DEBUG, logdir=logdir)
    # validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    validation_dataset = ValidationDataset(dataset_root_dir=dataset_root_dir, groups=val_dataset_groups,
                                           sequence_length=sequence_length, device=device, refresh=refresh, TOTAL_DEBUG=TOTAL_DEBUG, logdir=logdir)
    test_dataset = TestDataset(dataset_root_dir=dataset_root_dir, groups=test_dataset_groups,
                               sequence_length=sequence_length, device=device, refresh=refresh, TOTAL_DEBUG=TOTAL_DEBUG, logdir=logdir)

    loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(validation_dataset, 4,
                           shuffle=False, drop_last=True)
    # Getting one fixed batch for visualization
    batch_visualize = next(iter(valloader))

    if resume_iteration is None:
        ModelClass = create_model(model_type)
        model = ModelClass(ds_ksize=ds_ksize, ds_stride=ds_stride, reconstruction=reconstruction, mode=mode,
                           spec=spec, norm=sparsity, device=device, linear_head=linear_head, conv_head=conv_head, TOTAL_DEBUG=TOTAL_DEBUG, logdir=logdir)
        model.to(device)
        if pretrained_model_path != None:
            pretrained_model_path = pretrained_model_path
            print("Copying from ", pretrained_model_path)
            pretrained_model = torch.load(pretrained_model_path)
            model.load_my_state_dict(pretrained_model)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
            detected_epoch = detect_epoch(pretrained_model_path)
            optimizer.load_state_dict(torch.load(
                os.path.join(os.path.dirname(pretrained_model_path), f'last-optimizer-state-{detected_epoch}.pt')))
        else:
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
            resume_iteration = 0
        if freeze_all_layers:
            model.freeze_all_layers()
        if unfreeze_linear or unfreeze_lstm or unfreeze_conv:
            model.unfreeze_selected_layers(
                linear=unfreeze_linear, lstm=unfreeze_lstm, conv=unfreeze_conv)

    else:  # Loading checkpoints and continue training
        trained_dir = 'trained_MAPS'  # Assume that the checkpoint is in this folder
        model_path = os.path.join(trained_dir, f'{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(
            os.path.join(trained_dir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    # loop = tqdm(range(resume_iteration + 1, iterations + 1))
    total_batch = len(loader.dataset)
    # TO REPRODUCE:
    # python train.py with train_on=SynthesizedInstruments model_type=unet seed=33 pretrained_model_path=runs/TRAIN_TRANSCRIPTION_unet_ON_SynthesizedInstruments_CQT_imagewise_230324-203352/model-350.pt
    for ep in range(1, epoches+1):
        model.train()
        total_loss = 0
        batch_idx = 0
        # print(f'ep = {ep}, lr = {scheduler.get_lr()}')
        for batch in loader:
            predictions, losses, _ = model.run_on_batch(batch)
            clip_grad_norm_(model.parameters(), clip_gradient_norm)
            loss = sum(losses.values())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_idx += 1
            print(f'Train Epoch: {ep} [{batch_idx*batch_size}/{total_batch}'
                  f'({100. * batch_idx*batch_size / total_batch:.0f}%)]'
                  f'\tLoss: {loss.item():.6f}', end='\r')
        print(' '*100, end='\r')
        epoch_loss = total_loss/len(loader)
        print(f'Train Epoch: {ep}\tLoss: {epoch_loss:.6f}')
        fail_observer.snapshot.add_to_snapshot(ep,
                                               epoch_loss,
                                               f"Epoch: {ep} Loss: {epoch_loss}",
                                               copy.deepcopy(model.state_dict()),
                                               copy.deepcopy(optimizer.state_dict()),
                                               copy.deepcopy(scheduler.state_dict()))
        if fail_observer.snapshot.snapshot_triggered:
            print("The end of training! Snapshot is taken!")
            break
        # Logging results to tensorboard
        if ep == 1:
            #             os.makedirs(logdir, exist_ok=True) # creating the log dir
            writer = SummaryWriter(logdir)  # create tensorboard logger

        if (ep) % 10 == 0 and ep > 1:
            model.eval()
            with torch.no_grad():
                for key, values in evaluate_wo_velocity(validation_dataset, model, reconstruction=reconstruction).items():
                    if key.startswith('metric/'):
                        _, category, name = key.split('/')
                        print(
                            f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                        if ('precision' in name or 'recall' in name or 'f1' in name or 'levensthein' in name) and 'chroma' not in name:
                            writer.add_scalar(
                                key, np.mean(values), global_step=ep)

        if (ep) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(
                logdir, f'model-{ep}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(
                logdir, f'last-optimizer-state-{ep}.pt'))
        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=ep)

        # Load one batch from validation_dataset

        predictions, losses, mel = model.run_on_batch(batch_visualize)
        if ep == 1:  # Showing the original transcription and spectrograms
            fig, axs = plt.subplots(2, 2, figsize=(24, 8))
            axs = axs.flat
            for idx, i in enumerate(mel.cpu().detach().numpy()):
                axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Original', fig, ep)

            fig, axs = plt.subplots(2, 2, figsize=(24, 4))
            axs = axs.flat
            for idx, i in enumerate(batch_visualize['frame'].cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/Label', fig, ep)

        if ep < 11 or (ep % 50 == 0):
            fig, axs = plt.subplots(2, 2, figsize=(24, 4))
            axs = axs.flat
            for idx, i in enumerate(predictions['frame'].detach().cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/Transcription', fig, ep)

            fig, axs = plt.subplots(2, 2, figsize=(24, 8))
            if model_type != "resnet":
                axs = axs.flat
                for idx, i in enumerate(predictions['feat1'].detach().cpu().numpy()):
                    axs[idx].imshow(i[0].transpose(), cmap='jet',
                                    origin='lower', vmax=1, vmin=0)
                    axs[idx].axis('off')
                fig.tight_layout()
                writer.add_figure('images/feat1', fig, ep)
            if conv_head:
                fig, axs = plt.subplots(2, 2, figsize=(24, 8))
                axs = axs.flat
                for idx, i in enumerate(predictions['feat_conv'].detach().cpu().numpy()):
                    axs[idx].imshow(i[0].transpose(), cmap='jet',
                                    origin='lower', vmax=1, vmin=0)
                    axs[idx].axis('off')
                fig.tight_layout()
                writer.add_figure('images/feat_conv', fig, ep)

            if reconstruction:
                fig, axs = plt.subplots(2, 2, figsize=(24, 8))
                axs = axs.flat
                for idx, i in enumerate(predictions['feat2'].detach().cpu().numpy()):
                    axs[idx].imshow(i.transpose(), cmap='jet',
                                    origin='lower', vmax=1, vmin=0)
                    axs[idx].axis('off')
                fig.tight_layout()
                writer.add_figure('images/feat2', fig, ep)

                fig, axs = plt.subplots(2, 2, figsize=(24, 8))
                axs = axs.flat
                for idx, i in enumerate(predictions['feat1b'].detach().cpu().numpy()):
                    axs[idx].imshow(i[0].transpose(), cmap='jet',
                                    origin='lower', vmax=1, vmin=0)
                    axs[idx].axis('off')
                fig.tight_layout()
                writer.add_figure('images/feat1b', fig, ep)

                fig, axs = plt.subplots(2, 2, figsize=(24, 8))
                axs = axs.flat
                for idx, i in enumerate(predictions['reconstruction'].cpu().detach().numpy().squeeze(1)):
                    axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                    axs[idx].axis('off')
                fig.tight_layout()

                writer.add_figure('images/Reconstruction', fig, ep)

                fig, axs = plt.subplots(2, 2, figsize=(24, 4))
                axs = axs.flat
                for idx, i in enumerate(predictions['frame2'].detach().cpu().numpy()):
                    axs[idx].imshow(
                        i.transpose(), origin='lower', vmax=1, vmin=0)
                    axs[idx].axis('off')
                fig.tight_layout()
                writer.add_figure('images/Transcription2', fig, ep)

    # Evaluating model performance on the full MAPS songs in the test split
    print('Training finished, now evaluating ')
    with torch.no_grad():
        model = model.eval()
        metrics = evaluate_wo_velocity(tqdm(test_dataset), model, reconstruction=reconstruction,
                                       save_path=os.path.join(logdir, './MIDI_results'))

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(
                f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')

    export_path = os.path.join(logdir, 'result_dict')
    pickle.dump(metrics, open(export_path, 'wb'))
