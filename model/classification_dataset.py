from email.mime import audio
import json
import os
from abc import abstractmethod
from glob import glob
import sys
import pickle
from matplotlib.style import available
from torch.nn.functional import one_hot


import numpy as np
import soundfile
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import *
from .midi import parse_midi
import librosa

# torch.set_printoptions(profile="full")


def detect_classes(path):
    classes = []
    for path_instance in glob(path+"*"):
        class_name = path_instance.split("/")[-1]
        classes.append(class_name)
    return classes


def get_label_from_tsv_path(tsv_path, list_of_classess):
    for i, elem in enumerate(list_of_classess):
        if elem in tsv_path:
            return i
    return -1


class CalssificationDataset(Dataset):
    def __init__(self, path, dataset_root_dir=".", groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {self.path}")
        for group in groups:
            # self.files is defined in MAPS class
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                # self.load is a function defined below. It first loads all data into memory first
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])
        class_number = torch.LongTensor([data['label']])
        one_hot_encoded_label = one_hot(
            class_number, num_classes=self.number_of_classes)[0].type(torch.FloatTensor)
        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            if audio_length < self.sequence_length:
                number_of_missing_data = self.sequence_length // audio_length
                modulo_sequence_number = self.sequence_length % audio_length
                modulo_step_number = modulo_sequence_number // HOP_LENGTH
                result['audio'] = data['audio']
                for i in range(number_of_missing_data - 1):
                    result['audio'] = torch.cat(
                        (result['audio'], data['audio']))
                result['audio'] = torch.cat(
                    (result['audio'], data['audio'][:modulo_sequence_number])).to(self.device)
                result['label'] = one_hot_encoded_label.to(self.device)
            else:
                step_begin = self.random.randint(
                    audio_length - self.sequence_length) // HOP_LENGTH

                n_steps = self.sequence_length // HOP_LENGTH
                step_end = step_begin + n_steps

                begin = step_begin * HOP_LENGTH
                end = begin + self.sequence_length

                result['audio'] = data['audio'][begin:end].to(self.device)
                result['label'] = one_hot_encoded_label.to(
                    self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = one_hot_encoded_label.to(self.device)

        result['audio'] = result['audio'].float()
        return result

    def __len__(self):
        return len(self.data)

    @classmethod  # This one seems optional?
    @abstractmethod  # This is to make sure other subclasses also contain this method
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, instrument_label):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        audio = None
        saved_data_path = audio_path.replace(
            '.flac', '.pt').replace('.wav', '.pt')
        # Check if .pt files exist, if so just load the files
        if os.path.exists(saved_data_path) and self.refresh == False:
            loaded_data = torch.load(saved_data_path)
            audio = loaded_data['audio']
            label = instrument_label
            data = dict(path=audio_path, audio=audio, label=label)
            return data
        # Otherwise, create the .pt files
        if audio is None:
            audio, sr = librosa.load(
                audio_path, dtype='float32', mono=True, sr=SAMPLE_RATE)
            # = soundfile.read(audio_path, dtype='int16')
            assert sr == SAMPLE_RATE
            # convert numpy array to pytorch tensor
            audio = torch.FloatTensor(audio)
        label = instrument_label
        data = dict(path=audio_path, audio=audio, label=label)
        torch.save(data, saved_data_path)
        return data

    def get_number_of_instruments(self):
        return self.number_of_classes


class SynthesizedInstrumentsClassificationDataset(CalssificationDataset):

    def __init__(self, dataset_root_dir=".",  path='data/synthesize', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', classes=None):
        self.path = os.path.join(dataset_root_dir, path)
        if classes == None:
            self.detected_classes = detect_classes(self.path)
            self.number_of_classes = len(self.detected_classes)
        else:
            self.detected_classes = self.__perform_mapping_to_synthesized(classes)
            self.number_of_classes = len(self.detected_classes)
        super().__init__(self.path, dataset_root_dir, groups if groups is not None else [
            'all'], sequence_length, seed, refresh, device)

    def __perform_mapping_to_synthesized(self, classes_list):
        for i, elem in enumerate(classes_list):
            if elem == "MAPS":
                print("Mapped MAPS dataset to sytnthesized_acoustic_grand_piano")
                classes_list[i] = "synthesized_acoustic_grand_piano"
            if elem == "GuitarSet":
                print("Mapped GuitarSet dataset to sytnthesized_acoustic_grand_piano")
                classes_list[i] = "synthesized_acoustic_guitar_steel"
        return classes_list

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        # flacs = sorted(glob(os.path.join(self.path, "audio", '*solo*.wav')))
        # midis = sorted(glob(os.path.join(self.path, "labels", '*solo*.jams.mid')))
        flacs = []
        print(glob(self.path+"*"))
        if group in self.available_groups():
            for paths in glob(self.path+"*"):
                wav_files = os.path.join(paths, group, "audio", '*.wav')
                print(f"Adding: {wav_files}")
                flacs += sorted(glob(wav_files))
            print(f"Number of detected flacs / wavs: {len(flacs)}")
        if len(flacs) == 0:
            raise RuntimeError(f'Group {group} is empty')

        result = []
        print(f"Classes used for experiment: {self.detected_classes}")
        for audio_path in flacs:
            label = get_label_from_tsv_path(audio_path, self.detected_classes)
            result.append((audio_path, label))
        print(f"Combined data length: {len(result)}")
        return result

class MAPSClassificationDataset(CalssificationDataset):

    def __init__(self, dataset_root_dir=".", path="data/MAPS/", groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', classes=None):
        self.maps_path = path
        self.piano_class = 0
        if classes != None:
            if "synthesized_acoustic_grand_piano" in classes:
                self.piano_class = classes.index(
                    "synthesized_acoustic_grand_piano"
                )
            elif "MAPS" in classes:
                self.guitar_class = classes.index(
                    "MAPS")
            else:
                raise RuntimeError("Cannot map classes used to train model, to MAPS dataset!")
        else:
            self.piano_class = 0

        super().__init__('data/synthesize', dataset_root_dir, groups if groups is not None else [
            'AkPnBcht', 'AkPnBsdf'], sequence_length, seed, refresh, device)
        self.detected_classes = classes
        self.number_of_classes = len(self.detected_classes)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.maps_path, 'flac', '*_%s.flac' % group))
        class_indicator = [self.piano_class for f in flacs]
        #tsvs = [f.replace('/flac/', '/tsvs/').replace('.flac', '.tsv') for f in flacs]
        assert(all(os.path.isfile(flac) for flac in flacs))

        zipped_sorted_data = sorted(zip(flacs, class_indicator))
        return zipped_sorted_data


class GuitarSetClassificationDataset(CalssificationDataset):

    def __init__(self, dataset_root_dir=".", path='data/guitarset', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', classes=None):
        self.guitar_set_path = path
        self.guitar_class = 0
        if classes != None:
            if "synthesized_acoustic_guitar_steel" in classes:
                self.guitar_class = classes.index(
                    "synthesized_acoustic_guitar_steel")
            elif "GuitarSet" in classes:
                self.guitar_class = classes.index(
                    "GuitarSet")
            else:
                raise RuntimeError("Cannot map classes used to train model, to GuitarSet dataset!")
        else:
            self.guitar_class = 1
        super().__init__('data/synthesize', dataset_root_dir, groups if groups is not None else [
            'all'], sequence_length, seed, refresh, device)
        self.detected_classes = classes
        self.number_of_classes = len(self.detected_classes)

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        # flacs = sorted(glob(os.path.join(self.path, "audio", '*solo*.wav')))
        # midis = sorted(glob(os.path.join(self.path, "labels", '*solo*.jams.mid')))
        if group in self.available_groups():
            flacs = sorted(
                glob(os.path.join(self.guitar_set_path, group, "audio", '*.wav')))
            class_indicator = [self.guitar_class for f in flacs]
        if len(flacs) == 0:
            raise RuntimeError(f'Group {group} is empty')

        zipped_sorted_data = sorted(zip(flacs, class_indicator))
        return zipped_sorted_data


class MAPSandGuitarSetClassificationDataset(CalssificationDataset):
    def __init__(self, dataset_root_dir=".", groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', classes=None):
        self.guitar_set_path = os.path.join(dataset_root_dir,'data', 'guitarset')
        self.maps_path = os.path.join(dataset_root_dir,'data', 'MAPS')
        self.piano_class = 0
        self.guitar_class = 1
        self.maps_group_mapping = {
            "train": ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
            "val": ['ENSTDkAm', 'ENSTDkCl'],
            "test": ['ENSTDkAm', 'ENSTDkCl']
        }

        if classes != None:
            if "synthesized_acoustic_guitar_steel" in classes:
                self.guitar_class = classes.index(
                    "synthesized_acoustic_guitar_steel")
            elif "GuitarSet" in classes:
                self.guitar_class = classes.index(
                    "GuitarSet")
            if "synthesized_acoustic_grand_piano" in classes:
                self.piano_class = classes.index(
                    "synthesized_acoustic_grand_piano"
                )
            elif "MAPS" in classes:
                self.piano_class = classes.index(
                    "MAPS")
        super().__init__(path=dataset_root_dir,
                         dataset_root_dir=dataset_root_dir,
                         groups=groups if groups is not None else ['all'],
                         sequence_length=sequence_length,
                         seed=seed,
                         refresh=refresh,
                         device=device)
        self.detected_classes = ["MAPS", "GuitarSet"]
        self.number_of_classes = len(self.detected_classes)

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        files = []
        print(f"Scanning {self.guitar_set_path} and {self.maps_path}")
        if group in self.available_groups():
            guitar_set_wavs = sorted(
                glob(os.path.join(self.guitar_set_path, group, "audio", '*.wav')))
            guitarset_class_indicator = [self.guitar_class for f in guitar_set_wavs]
            print(f"Number of guitarset samples: {len(guitarset_class_indicator)}")
            files.extend(sorted(zip(guitar_set_wavs, guitarset_class_indicator)))
            for maps_group in self.maps_group_mapping[group]:
                maps_flacs = sorted(glob(os.path.join(self.maps_path, 'flac', '*_%s.flac' % maps_group)))
                maps_class_indicator = [self.piano_class for f in maps_flacs]
                files.extend(sorted(zip(maps_flacs, maps_class_indicator)))
            print(f"Number of MAPS samples : {len(files)-len(guitarset_class_indicator)}")
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')
        return files
# if __name__ == '__main__':
#     dataset = SynthesizedInstrumentsClassificationDataset(path='data/synthesize', dataset_root_dir="/home/common/datasets/amt/", groups=[
#         "train"], sequence_length=32000, seed=42, refresh=False, device='cpu')
#     loader = DataLoader(dataset, 32, shuffle=True, drop_last=True)
#     for batch in loader:
#         print(batch)
#     # detected_classes = detect_classes(
#     #     "/home/common/datasets/amt/data/synthesize")
#     # label = get_label_from_tsv_path("synthesize_rock_organ", detected_classes)
#     # print(label)
