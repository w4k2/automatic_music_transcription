import json
import os
from abc import abstractmethod
import glob
import pickle

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from .constants import *
from .midi import parse_midi
from .dataset_utils import *
import librosa


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, dataset_root_dir=".", groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        self.path = os.path.join(dataset_root_dir, path)
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

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            label_length = len(data['label'])
            if audio_length < self.sequence_length:
                destined_steps = self.sequence_length // HOP_LENGTH
                number_of_missing_audio = self.sequence_length // audio_length
                number_of_missing_label = destined_steps // label_length
                modulo_sequence_number = self.sequence_length % audio_length
                modulo_step_number = destined_steps - label_length*number_of_missing_label
                result['audio'] = data['audio'].to(self.device)
                result['label'] = data['label'].to(self.device)
                result['velocity'] = data['velocity'].to(self.device)
                #print(f"Original shapes: audio: {result['audio'].shape}, label: {result['label'].shape} velocity: {result['velocity'].shape}")
                #print(f"Number of missing data: {number_of_missing_audio}, number of missing label {number_of_missing_label} modulo sequence: {modulo_sequence_number}, destined_steps {destined_steps} modulo step: {modulo_step_number}")
                for i in range(number_of_missing_audio - 1):
                    result['audio'] = torch.cat((result['audio'], data['audio'].to(self.device))).to(self.device)
                for i in range(number_of_missing_label - 1):
                    result['label'] = torch.cat((result['label'], data['label'].to(self.device))).to(self.device)
                    result['velocity'] = torch.cat((result['velocity'], data['velocity'].to(self.device))).to(self.device)
                if(modulo_sequence_number > 0):
                    result['audio'] = torch.cat((result['audio'], data['audio'][:modulo_sequence_number].to(self.device))).to(self.device)
                if(modulo_step_number > 0):
                    result['label'] = torch.cat((result['label'], data['label'][:modulo_step_number].to(self.device))).to(self.device)
                    result['velocity'] = torch.cat((result['velocity'], data['velocity'][:modulo_step_number].to(self.device))).to(self.device)
                #print(f"Resulting shapes: audio: {result['audio'].shape}, label: {result['label'].shape} velocity: {result['velocity'].shape}")
            else:
                #print("DEBUG - overdriven audio length detected")
                step_begin = self.random.randint(
                    audio_length - self.sequence_length) // HOP_LENGTH

                n_steps = self.sequence_length // HOP_LENGTH
                step_end = step_begin + n_steps

                begin = step_begin * HOP_LENGTH
                end = begin + self.sequence_length

                result['audio'] = data['audio'][begin:end].to(self.device)
                result['label'] = data['label'][step_begin:step_end, :].to(
                    self.device)
                result['velocity'] = data['velocity'][step_begin:step_end, :].to(
                    self.device)
        else:
            #print("DEBUG - no sequence length detected")
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        # converting to float
        result['audio'] = result['audio'].to(self.device).float()
        result['onset'] = (result['label'] == 3).to(self.device).float()
        result['offset'] = (result['label'] == 1).to(self.device).float()
        result['frame'] = (result['label'] > 1).to(self.device).float()
        result['velocity'] = result['velocity'].to(self.device).float()
        # print(f"result['audio'].shape = {result['audio'].shape}")
        # print(f"result['label'].shape = {result['label'].shape}")
        #print(f"DEBUG - path: {result['path']} audio shape: {result['audio'].shape} label shape: {result['label'].shape}")
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

    def load(self, audio_path, tsv_path):
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
        saved_data_path = audio_path.replace(
            '.flac', '.pt').replace('.wav', '.pt')
        # Check if .pt files exist, if so just load the files
        if os.path.exists(saved_data_path) and self.refresh == False:
            return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio, sr = librosa.load(
            audio_path, dtype='float32', mono=True, sr=SAMPLE_RATE)
        # = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE
        # convert numpy array to pytorch tensor
        audio = torch.FloatTensor(audio)

        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        # This will affect the labels time steps
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        #if label extension is tsv
        label_extension = tsv_path.split(".")[-1]
        if label_extension == "tsv":
            universal_pianoroll_representation = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
            if not isinstance(universal_pianoroll_representation[0], np.ndarray):
                universal_pianoroll_representation = [universal_pianoroll_representation]
        elif label_extension == "mid" or label_extension == "midi":
            universal_pianoroll_representation = parse_midi(tsv_path)

        for onset, offset, note, vel in universal_pianoroll_representation:
            # Convert time to time step
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            # Ensure the time step of onset would not exceed the last time step
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            # Ensure the time step of frame would not exceed the last time step
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio,
                    label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, dataset_root_dir=".", path='MAESTRO/', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'train'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob.glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob.glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob.glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(
                open(os.path.join(self.path, 'maestro-v2.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace(
                '.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace(
                '.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f',
                           delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class GuitarSet(PianoRollAudioDataset):

    def __init__(self, dataset_root_dir=".", path='data/guitarset', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'all'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        print(f"Looking for data in path {self.path} for group {group}")
        # flacs = sorted(glob.glob(os.path.join(self.path, "audio", '*solo*.wav')))
        # midis = sorted(glob.glob(os.path.join(self.path, "labels", '*solo*.jams.mid')))
        if group in self.available_groups():
            flacs = sorted(
                glob.glob(os.path.join(self.path, group, "audio", '*.wav')))
            midis = sorted(
                glob.glob(os.path.join(self.path, group, "labels", '*.mid')))
            files = list(zip(flacs, midis))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')

        result = []
        for audio_path, midi_path in files:
            midi = parse_midi(midi_path)
            tsv_filename = midi_path+".tsv"
            np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t',
                       header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class SynthesizedTrumpet(PianoRollAudioDataset):

    def __init__(self, dataset_root_dir=".", path='data/synthesize_trumpet', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'all'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        # flacs = sorted(glob.glob(os.path.join(self.path, "audio", '*solo*.wav')))
        # midis = sorted(glob.glob(os.path.join(self.path, "labels", '*solo*.jams.mid')))
        if group in self.available_groups():
            flacs = sorted(
                glob.glob(os.path.join(self.path, group, "audio", '*.wav')))
            midis = sorted(
                glob.glob(os.path.join(self.path, group, "labels", '*.mid')))
            files = list(zip(flacs, midis))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path+".tsv"
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t',
                        header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result

class SynthesizedInstruments(PianoRollAudioDataset):

    def __init__(self, dataset_root_dir=".",  path='data/synthesize', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'all'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        # flacs = sorted(glob.glob(os.path.join(self.path, "audio", '*solo*.wav')))
        # midis = sorted(glob.glob(os.path.join(self.path, "labels", '*solo*.jams.mid')))
        wavs = []
        labels = []
        if group in self.available_groups():
            for paths in glob.glob(self.path+"*"):
                print(f"Adding instrument from {paths} to {group}")
                found_wavs = sorted(
                    glob.glob(os.path.join(paths, group, "audio", '*.wav')))
                for wav in found_wavs:
                    label = find_label_for_given_wav(os.path.join(paths, group, "labels"), wav)
                    if label == None:
                        print(f"Warning - couldn't find corresponding label for file {wav}")
                        assert(False)
                    else:
                        labels.append(label)
                        wavs.append(wav)
                check_consistency_of_size_of_audio_and_labels(wavs, labels)
            files = list(zip(wavs, labels))
            print(f"Number of audio samples: {len(wavs)}, Number of labels: {len(labels)}")
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')

        return prepare_list_of_tuples_with_audio_and_label_filenames(files)


class MAPS(PianoRollAudioDataset):
    def __init__(self, dataset_root_dir=".",  path='data/MAPS', groups=None, sequence_length=None, overlap=True, seed=42, refresh=False, device='cpu'):
        self.overlap = overlap
        print("Initializing MAPS dataset!")
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob.glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        if self.overlap == False:
            with open('overlapping.pkl', 'rb') as f:
                test_names = pickle.load(f)
            filtered_flacs = []
            for i in flacs:
                if any([substring in i for substring in test_names]):
                    pass
                else:
                    filtered_flacs.append(i)
            flacs = filtered_flacs
        tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv')
                for f in flacs]
        #tsvs = [f.replace('/flac/', '/tsvs/').replace('.flac', '.tsv') for f in flacs]
        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))
        assert(len(flacs) == len(tsvs))

        return sorted(zip(flacs, tsvs))


class OriginalMAPS(PianoRollAudioDataset):
    def __init__(self, dataset_root_dir=".",  path='MAPS', groups=None, sequence_length=None, overlap=True, seed=42, refresh=False, device='cpu'):
        self.overlap = overlap
        print(f"Initializing original MAPS dataset with {groups}!")
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        if group in self.available_groups():
            path_to_wavs_list = os.path.join(self.path, f"list_of_maps_wavs_{group}.np")
            if(not os.path.exists(path_to_wavs_list) or self.refresh == True):
                list_of_wavs = glob.glob(self.path+"/**/"+ f'*{group}.wav', recursive=True)
                list_of_wavs = list(filter(lambda elem: "MUS" in elem, list_of_wavs))
                np.save(path_to_wavs_list, list_of_wavs)
            else:
                print(f"Loading existing wav filelist for {group} group!")
                list_of_wavs = np.load(path_to_wavs_list)
            path_to_midis_list = os.path.join(self.path, f"list_of_maps_midis_{group}.np")
            if(not os.path.exists(path_to_midis_list) or self.refresh == True):
                list_of_midis = glob.glob(self.path+"/**/"+ f'*{group}.mid', recursive=True)
                list_of_midis = list(filter(lambda elem: "MUS" in elem, list_of_midis))
                np.save(path_to_midis_list, list_of_midis)
            else:
                print(f"Loading existing midi filelist for {group} group!")
                list_of_midis = np.load(path_to_midis_list)

            flacs = sorted(list_of_wavs)
            midis = sorted(list_of_midis)
            files = list(zip(flacs, midis))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')

        result = []
        for audio_path, midi_path in files:
            midi = parse_midi(midi_path)
            tsv_filename = midi_path+".tsv"
            np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t',
                       header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result

class CustomBatchDataset(Dataset):
    def __init__(self, dataset_root_dir=".", path='batch/', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        self.device = device
        directories_with_batch_data = glob.glob(os.path.join(dataset_root_dir, "*"))
        self.data = []
        for epoch_data in directories_with_batch_data:
            directories_per_batch = glob.glob(os.path.join(epoch_data, "*"))
            for directory in tqdm(directories_per_batch, desc=f"Loading batch from {epoch_data}"):
                self.data.append(self.load(directory))
        self.last_visited_index = 0
        self.size = len(self.data)
        self.dataset = self.data

    @classmethod
    def available_groups(cls):
        return ['test']

    def load(self, directory):
        audio = np.load(os.path.join(directory, "batch_audio.npy"))
        audio = torch.FloatTensor(audio).to(self.device).float()
        frame = np.load(os.path.join(directory, "batch_frame.npy"))
        frame = torch.ByteTensor(frame).to(self.device).float()
        path = np.load(os.path.join(directory, "batch_path.npy")).tolist()
        data = dict(path=path, audio=audio,
                    frame=frame)
        return data

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

class MusicNet(PianoRollAudioDataset):
    def __init__(self, dataset_root_dir=".", path='MusicNet/', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'train'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'test']

    def files(self, group):
        if group == 'small test':
            types = ('2303.flac', '2382.flac', '1819.flac')
            flacs = []
            for i in types:
                flacs.extend(glob.glob(os.path.join(self.path, 'test_data', i)))
            flacs = sorted(flacs)
            tsvs = sorted(
                glob.glob(os.path.join(self.path, f'tsv_test_labels/*.tsv')))
        else:
            flacs = sorted(
                glob.glob(os.path.join(self.path, f'{group}_data/*.flac')))
            tsvs = sorted(
                glob.glob(os.path.join(self.path, f'tsv_{group}_labels/*.tsv')))

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return zip(flacs, tsvs)
