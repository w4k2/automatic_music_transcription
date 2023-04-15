#!/bin/bash

function evaluate_model {
    python evaluate.py with weight_file=$1 dataset=GuitarSet
    python evaluate.py with weight_file=$1 dataset=MAPS
    python evaluate.py with weight_file=$1 dataset=SynthesizedInstruments
}
epoches=2000
seed=33
#BASIC TRANSCRIPTION
python train.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS model_type=unet epoches=$epoches seed=33
python train.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet model_type=unet epoches=$epoches seed=33
python train.py with train_on=SynthesizedInstruments logdir=results/unet_model_trained_on_SynthesizedInstruments model_type=unet epoches=$epoches seed=33

#EVALUATE TRANSCRIPTION
evaluate_model results/unet_model_trained_on_MAPS/model-$epoches.pt
evaluate_model results/unet_model_trained_on_GuitarSet/model-$epoches.pt
evaluate_model results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt

#TRANSFER FROM SYNTHESIZED INSTRUMENTS
python train.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=34
python train.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=34

#EVALUATE TRANSFER
evaluate_model results/transferred_unet_model_trained_on_MAPS/model-$epoches.pt
evaluate_model results/transferred_unet_model_trained_on_GuitarSet/model-$epoches.pt

#TRANSFER FROM GUITARSET
python train.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=34
evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS/model-$epoches.pt

#TRANSFER FROM MAPS
python train.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=34
evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet/model-$epoches.pt

python result_dict_analysis.py

python result_table_generator.py results/unet_model_trained_on_MAPS results/unet_model_trained_on_GuitarSet results/unet_model_trained_on_SynthesizedInstruments  results/transferred_unet_model_trained_on_MAPS results/transferred_unet_model_trained_on_GuitarSet results/transferred_from_guitarset_unet_model_trained_on_MAPS results/transferred_from_MAPS_unet_model_trained_on_GuitarSet > results/table.txt
