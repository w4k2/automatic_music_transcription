import os


from datetime import datetime
import pickle
import torch

import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.classification_dataset import SynthesizedInstrumentsClassificationDataset, MAPSClassificationDataset, GuitarSetClassificationDataset, MAPSandGuitarSetClassificationDataset
from model.dataset import SynthesizedInstruments
from model.instrument_recognition_model import InstrumentRecognitionModel

from model.evaluate_fn import evaluate_classification
from model import *
from classesstxt_utils import save_classes_to_file, load_model_type_from_directory, save_model_type_to_file, load_classess

import matplotlib.pyplot as plt
#import wandb

# wandb.init(project="amt-test")

#set random seed for whole experiment
torch.manual_seed(33)
import random
random.seed(33)
np.random.seed(33)

ex = Experiment('train_transcriber')

# parameters for the network
ds_ksize, ds_stride = (2, 2), (2, 2)
mode = 'imagewise'
sparsity = 1


@ex.config
def config():
    logdir = f'runs_AE/test' + '-' + datetime.now().strftime('%y%m%d-%H%M%S')
    # Choosing GPU to use
    GPU = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    device = 'cuda'
    dataset_root_dir = "."
    spec = 'CQT'
    transfer_from = None
    resume_iteration=200

    batch_size = 64
    sequence_length = 327680

    epoches = 200
    learning_rate = 0.01
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98
    model_type = "resnet"
    dataset_name = "AllSynthesizedInstruments"
    custom_output_dir = "runs"

    clip_gradient_norm = 3
    if(transfer_from!=None):
        dataset_used_for_training = transfer_from.split("/")[-1].split("_")[2]
        model_used_for_training = transfer_from.split("/")[-1].split("_")[3]
        transfer_info = f"TRANSFER_FROM_{dataset_used_for_training}_{model_used_for_training}_"
    else:
        transfer_info="NO_TRANSFER_"
    logdir = f'{custom_output_dir}/TRAIN_CLASSIFICATION_{dataset_name}_{model_type}_{transfer_info}' + \
        datetime.now().strftime('%y%m%d-%H%M%S')

    ex.observers.append(FileStorageObserver.create(logdir)
                        )  # saving source code

def dataset_factory(dataset_name):
    if dataset_name == "MAPS":
        print("############ MAPS dataset detected!")
        return MAPSClassificationDataset, (['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], ['ENSTDkAm', 'ENSTDkCl'], ['ENSTDkAm', 'ENSTDkCl'])
    elif dataset_name == "AllSynthesizedInstruments":
        print("############ Synthesized dataset detected!")
        return SynthesizedInstrumentsClassificationDataset, (["train"], ["val"], ["test"])
    elif dataset_name == "GuitarSet":
        print("############ GuitarSet dataset detected!")
        return GuitarSetClassificationDataset, (["train"], ["val"], ["test"])
    elif dataset_name == "MixedMapsAndGuitarSet":
        print("############ MAPS and GuitarSet mixed dataset detected!")
        return MAPSandGuitarSetClassificationDataset, (["train"], ["val"], ["test"])
    raise Exception("FATAL ERROR - no proper dataset detected!!!")

@ex.automain
def train(spec, transfer_from, resume_iteration, batch_size, sequence_length, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate,
          clip_gradient_norm, device, epoches, logdir, dataset_root_dir, model_type, dataset_name):
    print_config(ex.current_run)

    # wandb.config = {
    #     "learning_rate": learning_rate,
    #     "epochs": epoches,
    #     "batch_size": batch_size
    # }
    # Choosing the dataset to use
    dataset =  dataset_factory(dataset_name)
    DatasetClass = dataset[0]
    train_groups = dataset[1][0]
    val_groups = dataset[1][1]
    test_groups = dataset[1][2]
    
    dataset = DatasetClass(dataset_root_dir=dataset_root_dir, groups=train_groups, sequence_length=sequence_length, device=device)

    validation_dataset = DatasetClass(dataset_root_dir=dataset_root_dir, groups=val_groups, sequence_length=sequence_length, device=device)

    test_dataset = DatasetClass(dataset_root_dir=dataset_root_dir, groups=test_groups, sequence_length=sequence_length, device=device)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(validation_dataset, 4,
                           shuffle=False, drop_last=True)
    testloader = DataLoader(test_dataset, 4, shuffle=False, drop_last=True)

    detected_classes = dataset.detected_classes
    print("Detected classes: ", detected_classes)
    save_classes_to_file(logdir, detected_classes)
    save_model_type_to_file(logdir, model_type)

    if (transfer_from != None):
        pretrained_model_path = transfer_from
        print(f"Transferring all weights from {pretrained_model_path}")
        detected_model = load_model_type_from_directory(pretrained_model_path)
        print(f"Detected model : {detected_model}")
        original_classes = load_classess(pretrained_model_path)
        print(f"Original model classes: {original_classes}")
        model = InstrumentRecognitionModel(ds_ksize, ds_stride, mode=mode,
                                        spec=spec, norm=sparsity, device=device, number_of_instruments=len(original_classes), model_type=detected_model)
        model.to(device)
        model_path = os.path.join(pretrained_model_path, f'model-{resume_iteration}.pt')
        model_state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_my_state_dict(state_dict=model_state_dict,
                                 blacklist=["classification_layer"])
    else:
        model = InstrumentRecognitionModel(ds_ksize, ds_stride, mode=mode,
                                       spec=spec, norm=sparsity, device=device, number_of_instruments=dataset.get_number_of_instruments(), model_type=model_type)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    summary(model)
    scheduler = StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    # loop = tqdm(range(resume_iteration + 1, iterations + 1))
    total_batch = len(loader.dataset)
    for ep in range(1, epoches+1):
        model.train()
        total_loss = 0
        batch_idx = 0
        # print(f'ep = {ep}, lr = {scheduler.get_lr()}')
        for batch in loader:
            predictions, losses, _ = model.run_on_batch(batch)
            loss = sum(losses.values())
            # wandb.log({"loss": loss})
            # wandb.watch(model)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            clip_grad_norm_(model.parameters(), clip_gradient_norm)
            batch_idx += 1
            print(f'Train Epoch: {ep} [{batch_idx*batch_size}/{total_batch}'
                  f'({100. * batch_idx*batch_size / total_batch:.0f}%)]'
                  f'\tLoss: {loss.item():.6f}', end='\r')
        print(' '*100, end='\r')
        print(f'Train Epoch: {ep}\tLoss: {total_loss/len(loader):.6f}')
        if ep == 1:
            #             os.makedirs(logdir, exist_ok=True) # creating the log dir
            writer = SummaryWriter(logdir)  # create tensorboard logger

        if (ep) % 10 == 0 and ep > 1:
            model.eval()
            with torch.no_grad():
                for key, values in evaluate_classification(valloader, model, batch_size).items():
                    if key.startswith('metrics/'):
                        _, category, name = key.split('/')
                        print(
                            f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                        writer.add_scalar(
                            key, np.mean(values), global_step=ep)
        if (ep) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(
                logdir, f'model-{ep}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(
                logdir, 'last-optimizer-state.pt'))
        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=ep)

    model.eval()
    with torch.no_grad():
        for key, values in evaluate_classification(testloader, model, batch_size).items():
            if key.startswith('metrics/'):
                _, category, name = key.split('/')
                print(
                    f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                writer.add_scalar(
                    key, np.mean(values), global_step=ep)
