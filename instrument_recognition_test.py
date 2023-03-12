import os


from datetime import datetime
import torch

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from model.instrument_recognition_model import InstrumentRecognitionModel
from model.evaluate_fn import evaluate_classification
from model import *
from classesstxt_utils import load_classess, load_model_type_from_directory
from instrument_recognition_train import dataset_factory


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
    # Choosing GPU to use
    GPU = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

    spec = 'CQT'
    batch_size = 64
    sequence_length = 327680
    device = 'cuda'
    dataset_root_dir = "."
    pretrained_model_path = None
    resume_iteration = 200
    dataset_name = "GuitarSet"
    dataset_used_for_training = pretrained_model_path.split("/")[-1].split("_")[2]
    model_used_for_training = pretrained_model_path.split("/")[-1].split("_")[3]
    if("NO_TRANSFER" in pretrained_model_path):
        transfer_info=""
    else:
        transfer_info="TRANSFER_"
    custom_output_dir = "test_runs"
    logdir = f'{custom_output_dir}/TEST_CLASSIFICATION_{dataset_used_for_training}_EPOCH{resume_iteration}_{model_used_for_training}_{transfer_info}ON_{dataset_name}_' + \
        datetime.now().strftime('%y%m%d-%H%M%S')

    ex.observers.append(FileStorageObserver.create(logdir)
                        )  # saving source code


@ex.automain
def test(spec, pretrained_model_path, batch_size, sequence_length,
         device, dataset_root_dir, resume_iteration, dataset_name):
    print_config(ex.current_run)

    original_classes = load_classess(pretrained_model_path)
    print(f"Original model classes: {original_classes}")

    dataset =  dataset_factory(dataset_name)
    DatasetClass = dataset[0]
    test_groups = dataset[1][2]

    test_dataset = DatasetClass(dataset_root_dir=dataset_root_dir,
                                groups=test_groups,
                                sequence_length=sequence_length,
                                device=device,
                                classes=original_classes)
    testloader = DataLoader(test_dataset, 4, shuffle=False, drop_last=True)

    trained_dir = pretrained_model_path
    print(f"Transferring all weights from {pretrained_model_path}")

    detected_model = load_model_type_from_directory(trained_dir)
    print(f"Detected model : {detected_model}")
    model = InstrumentRecognitionModel(ds_ksize, ds_stride, mode=mode,
                                       spec=spec, norm=sparsity, device=device, number_of_instruments=len(original_classes), model_type=detected_model)
    model.to(device)
    model_path = os.path.join(trained_dir, f'model-{resume_iteration}.pt')
    model_state_dict = torch.load(
        model_path, map_location=torch.device(device))
    model.load_my_state_dict(model_state_dict)

    summary(model)

    model.eval()
    with torch.no_grad():
        for key, values in evaluate_classification(testloader, model, batch_size, classes=original_classes).items():
            if key.startswith('metrics/'):
                _, category, name = key.split('/')
                print(
                    f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
