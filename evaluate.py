import os


from datetime import datetime
import pickle

import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from train import create_transcription_datasets
import json


from model.evaluate_fn import evaluate_wo_velocity
from model import *

import matplotlib.pyplot as plt
ex = Experiment('Evaluation')

# parameters for the network (These parameters works the best)
ds_ksize, ds_stride = (2, 2), (2, 2)
mode = 'imagewise'
sparsity = 1
log = True  # Turn on log magnitude scale spectrograms.


def removing_nnAudio_parameters(state_dict):
    pop_list = []
    for i in state_dict.keys():
        if i.startswith('spectrogram'):
            pop_list.append(i)

    print(f'The following weights will be remove:\n{pop_list}')
    decision = input("Do you want to proceed? [y/n] ")

    while True:
        if decision.lower() == 'y':
            for i in pop_list:
                state_dict.pop(i)
            return state_dict
        elif decision.lower() == 'n':
            return state_dict

        print(f'Please choose only [y] or [n]')
        decision = input("Do you want to proceed? [y/n] ")


@ex.config
def config():
    weight_file = 'MAESTRO-CQT-transcriber_only'
    destination_dir = os.path.join(os.path.dirname(os.path.abspath(weight_file)), "eval")
    spec = "CQT"
    dataset = 'MAPS'
    logdir = os.path.join(destination_dir, f"VALIDATION_ON_{dataset}_OF_"+weight_file.split("/")[-2])
    device = 'cuda:0'
    reconstruction = False
    conv_head = False
    linear_head = True
    dataset_root_dir = "."
    leave_one_out = None
    refresh = False


@ex.automain
def evaluate(spec, dataset, device, reconstruction, logdir, leave_one_out, weight_file, conv_head, linear_head, dataset_root_dir, refresh):
    print_config(ex.current_run)
    dataset_data = create_transcription_datasets(dataset_type=dataset)
    TestDataset = dataset_data[2][0]
    test_dataset_groups = dataset_data[2][1]

    validation_dataset = TestDataset(dataset_root_dir=dataset_root_dir, groups=test_dataset_groups,
                                     sequence_length=327680, device=device, refresh=refresh)


    model = NetWithAdditionalHead(ds_ksize, ds_stride, log=log, reconstruction=reconstruction,
                                  mode=mode, spec=spec, norm=sparsity, linear_head=linear_head, conv_head=conv_head)
    model.to(device)
    model_path = weight_file
    state_dict = torch.load(model_path)
    model.load_my_state_dict(state_dict)

    summary(model)

    with torch.no_grad():
        model = model.eval()
        metrics = evaluate_wo_velocity(tqdm(validation_dataset), model, reconstruction=reconstruction,
                                       save_path=os.path.join(logdir, f'./{dataset}_MIDI_results'))

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(
                f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')

    export_path = os.path.join(logdir, f'{dataset}_result_dict')
    pickle.dump(metrics, open(export_path, 'wb'))
    import json
    with open(f'{export_path}.txt', 'w') as convert_file:
        convert_file.write(json.dumps(metrics))
