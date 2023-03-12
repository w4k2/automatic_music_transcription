# Automatic Music Transcription
This repository is based on the [repository](https://github.com/KinWaiCheuk/ICPR2020) prepared for the [paper](https://arxiv.org/abs/2010.09969) written by Kin Wai Cheuk, Yin-Jyun Luo, Emmanouil Benetosand Dorien Herremans

## Setup environment
To run all experiments from this project you need to use Python3.8. Because it may be not supported natively on some platforms and linux distributions you can use conda version of python3.8. Unfortunately currently conda does not support some important audio analysis packages, that's why I recommend to setup environment using virtualenv:
* check if your python version is 3.8 (it should start with 3.8):
    ```
    python3 --version
    ```
* check version of virtualenv - in theory it shouldn't make big difference:
    ```
    virtualenv --version
    ```
* setup environment using virtualenv:
    ```
    virtualenv -p python3.8 venv
    ```
* source into environment and perform requirements installation using pip:
    ```
    source venv/bin/activate
    pip install -r requirements.txt
    ```

Note - special version of pytorch needed for GPU support may be needed to be setup seperately, for example using following command:
```
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
For details you can visit official [pytorch site](https://pytorch.org/get-started/locally/).

Additionally - to successfully perform dataset creation on your local machine you must use FluidSynth software for music synthesis. It is most commonly distributed under `fluidsynth` package and you need to install it manually on your operating system. To check if fluidsynth is installed correctly you can use following command:
```
fluidsynth --version
```
You must also download some music font for existing FluidSynth distribution and place it in your local home directory (`~/.fluidsynth/default_sound_font.sf2`). For this project I used [FluidR3](https://member.keymusician.com/Member/FluidR3_GM/index.html) soundfont.


## Preparing datasets
Each dataset may have specific rules of creation and maintanance.
* `MAPS` - is is the most popular dataset for automatic music transcription. Is embedded into this project in `data` directory. It is important to mention, that it is modified version (changed format of audio data) and `data/MAPS` is shared under [non-comercial Creative Commons license](https://creativecommons.org/licenses/by-nc-sa/2.0/fr/deed.en_US). It is ready to use.
* `GuitarSet` - it is dataset containing recording of guitar music and transcription in jams format.. To use it in project you need to download it. There is jupyter notebook called `prepare_guitarset.ipynb` located in `data` directory, which will help you with download, extraction and preparation of GuitarSet data to be used in the project.
* `SynthesizedInstruments` - this dataset needs to be created using methods for synthesized music creation. It is created by taking all transcriptions for `MAPS` and `GuitarSet` and making music from them using different synthesized instruments. Number of instruments is not limited. After choosing this option the program will look for all instruments stored in starting with word `synthesize*` in `data` directory of `dataaset_root_dir`.   
**WARNING! you need to have valid `GuitarSet` and `MAPS` dataset before generation of synthesized instruments!**  
To generate synthesized datasets you need to run `prepare_synthesized_datasets_script.py` and manually modify list of instruments:
    ```
    python prepare_synthesized_datasets_script.py
    ```

## Training the transcription model
The python script can be run using using the sacred syntax `with`.
```python
python train.py with train_on=<arg> spec=<arg> device=<arg> destination_dir=<arg> logdir=<arg> refresh=<arg> dataset_root_dir=<arg> model_type=<arg> pretrained_model_path=<arg>
```

* `train_on`: the dataset to be trained on. Either `MAPS` or `GuitarSet` or `SynthesizedInstruments`
* `spec`: the input spectrogram type. Either `Mel` or `CQT` (default `CQT`).
* `device`: the device to be trained on. Either `cpu` or `cuda:0`
* `destination_dir`: specify destination directory for all outputs for training using default description method (default destination directory is `runs`)
* `logdir`: specify special name for log directory - overrides destination dir, use only when you want to manually specify deterministic (no date-dependent) output directory
* `refresh`: specify if spectrograms needs to be refreshed for dataset. If set to `True` then preprocessed data will be used. It will slow down data preparation phase of training (default: `False`).
* `dataset_root_dir`: use different directory as a base for datasets (`data` directory). Useful for different bases for synthesized instruments. Default value is current directory (`./`)
* `model_type`: using this parameter model type may be changed. Default model for transcription is `unet` and it is the main part of this experiment. There is experimental support for `resnet` model, but it appears to not work very well for transcription.
* `pretrained_model_path`: path to weight file (with `.pt` extension) containing pretrained model - transfers all weights from this model at the beginning of training.
* `other params`: there are a lot of params used for model customization and possible variations. My recommendation is to leave it with default values. Most of them are not needed and will be removed in the future.

## Evaluating the model and exporting the midi files

```python
python evaluate.py with weight_file=<arg> device=<arg>
```

* `weight_file`: path to file with `.pt` extension containing dictionary with model parameters. in directory containing this model `eval` directory will be created, containing results of evaluation.
* `dataset`: which dataset to evaluate on, can be either `MAPS` or `GuitarSet` or `SynthesizedInstruments`.
* `dataset_root_dir`: use different directory as a base for datasets (`data` directory). Useful for different bases for synthesized instruments.
* `device`: the device to be trained on. Either `cpu` or `cuda:0`

To calculate average results for all evaluated models, after evaluation of all of them you should run `result_dict_analysis.py` script.