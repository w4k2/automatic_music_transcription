#!/bin/bash

function evaluate_model {
    python evaluate.py with weight_file=$1 dataset=GuitarSet
    python evaluate.py with weight_file=$1 dataset=MAPS
    python evaluate.py with weight_file=$1 dataset=SynthesizedInstruments dataset_root_dir=data/synthesized_piano_and_guitar
}
epoches=100
#BASIC TRANSCRIPTION
python train.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS model_type=unet refresh=True epoches=$epoches
python train.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet model_type=unet refresh=True epoches=$epoches
python train.py with train_on=SynthesizedInstruments logdir=results/unet_model_trained_on_SynthesizedInstruments model_type=unet epoches=$epoches

#EVALUATE TRANSCRIPTION
evaluate_model results/unet_model_trained_on_MAPS/model-$epoches.pt
evaluate_model results/unet_model_trained_on_GuitarSet/model-$epoches.pt
evaluate_model results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt

#TRANSFER FROM SYNTHESIZED INSTRUMENTS
python train.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches
python train.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches

#EVALUATE TRANSFER
evaluate_model results/transferred_unet_model_trained_on_MAPS/model-$epoches.pt
evaluate_model results/transferred_unet_model_trained_on_GuitarSet/model-$epoches.pt

python result_dict_analysis.py
