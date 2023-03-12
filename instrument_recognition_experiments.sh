#!/bin/bash
CUSTOM_OUTPUT_DIR="final_results"
function getPath
{
    file="$(realpath $1*)"
    echo $file
}

# #training and testing for synthesized piano and guitar
python instrument_recognition_train.py with custom_output_dir=$CUSTOM_OUTPUT_DIR dataset_root_dir=/home/common/datasets/amt/synthesized_piano_and_guitar/ dataset_name="AllSynthesizedInstruments" epoches=200

python instrument_recognition_test.py with custom_output_dir=$CUSTOM_OUTPUT_DIR pretrained_model_path=$(getPath $CUSTOM_OUTPUT_DIR/TRAIN_CLASSIFICATION_AllSynthesizedInstruments_resnet_NO_TRANSFER) dataset_root_dir=/home/common/datasets/amt/synthesized_piano_and_guitar/ dataset_name="AllSynthesizedInstruments" resume_iteration=200
python instrument_recognition_test.py with custom_output_dir=$CUSTOM_OUTPUT_DIR pretrained_model_path=$(getPath $CUSTOM_OUTPUT_DIR/TRAIN_CLASSIFICATION_AllSynthesizedInstruments_resnet_NO_TRANSFER) dataset_name="MixedMapsAndGuitarSet" resume_iteration=200

# # training for mixed MAPS and GuitarSet
python instrument_recognition_train.py with custom_output_dir=$CUSTOM_OUTPUT_DIR dataset_name="MixedMapsAndGuitarSet" epoches=200

python instrument_recognition_test.py with custom_output_dir=$CUSTOM_OUTPUT_DIR pretrained_model_path=$(getPath $CUSTOM_OUTPUT_DIR/TRAIN_CLASSIFICATION_MixedMapsAndGuitarSet_resnet_NO_TRANSFER) dataset_root_dir=/home/common/datasets/amt/synthesized_piano_and_guitar/ dataset_name="AllSynthesizedInstruments" resume_iteration=200
python instrument_recognition_test.py with custom_output_dir=$CUSTOM_OUTPUT_DIR pretrained_model_path=$(getPath $CUSTOM_OUTPUT_DIR/TRAIN_CLASSIFICATION_MixedMapsAndGuitarSet_resnet_NO_TRANSFER) dataset_name="MixedMapsAndGuitarSet" resume_iteration=200

# #training for mixed MAPS and GuitarSet with transferred knowledge from synthesized instruments
python instrument_recognition_train.py with custom_output_dir=$CUSTOM_OUTPUT_DIR dataset_name="MixedMapsAndGuitarSet" transfer_from=$(getPath $CUSTOM_OUTPUT_DIR/TRAIN_CLASSIFICATION_AllSynthesizedInstruments_resnet_NO_TRANSFER) epoches=200
python instrument_recognition_test.py with custom_output_dir=$CUSTOM_OUTPUT_DIR pretrained_model_path=$(getPath $CUSTOM_OUTPUT_DIR/TRAIN_CLASSIFICATION_MixedMapsAndGuitarSet_resnet_TRANSFER_FROM) dataset_root_dir=/home/common/datasets/amt/synthesized_piano_and_guitar/ dataset_name="AllSynthesizedInstruments" resume_iteration=200
python instrument_recognition_test.py with custom_output_dir=$CUSTOM_OUTPUT_DIR pretrained_model_path=$(getPath $CUSTOM_OUTPUT_DIR/TRAIN_CLASSIFICATION_MixedMapsAndGuitarSet_resnet_TRANSFER_FROM) dataset_name="MixedMapsAndGuitarSet" resume_iteration=200

