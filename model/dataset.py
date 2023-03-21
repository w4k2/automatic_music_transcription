from email.mime import audio
import json
import os
from abc import abstractmethod
from glob import glob
import sys
import pickle
from matplotlib.style import available
from .tensor_visualizer import TensorVisualizer


import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
from .constants import *
from .midi import parse_midi
import librosa

# torch.set_printoptions(profile="full")


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, dataset_root_dir=".", groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', TOTAL_DEBUG=False, logdir="runs/logdir"):
        self.TOTAL_DEBUG=TOTAL_DEBUG
        self.path = os.path.join(dataset_root_dir, path)
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh
        self.tensor_visualizer = TensorVisualizer(logdir)

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
        if self.TOTAL_DEBUG:
            copy_of_audio = result['audio'].clone().detach()
            copy_of_frame = result['frame'].clone().detach()
            self.tensor_visualizer.save_audio_from_pytorch_tensor(copy_of_audio, result['path'], index)
            self.tensor_visualizer.save_pianoroll_from_pytorch_tensor(copy_of_frame, result['path'], index)
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

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        if not isinstance(midi[0], np.ndarray):
            midi = [midi]


        for onset, offset, note, vel in midi:
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

    def __init__(self, dataset_root_dir=".", path='MAESTRO/', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', TOTAL_DEBUG=False, logdir="runs/logdir"):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'train'], sequence_length, seed, refresh, device, TOTAL_DEBUG, logdir)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
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

    def __init__(self, dataset_root_dir=".", path='data/guitarset', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', TOTAL_DEBUG=False, logdir="runs/logdir"):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'all'], sequence_length, seed, refresh, device, TOTAL_DEBUG, logdir)

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        print(f"Looking for data in path {self.path} for group {group}")
        # flacs = sorted(glob(os.path.join(self.path, "audio", '*solo*.wav')))
        # midis = sorted(glob(os.path.join(self.path, "labels", '*solo*.jams.mid')))
        if group in self.available_groups():
            flacs = sorted(
                glob(os.path.join(self.path, group, "audio", '*.wav')))
            midis = sorted(
                glob(os.path.join(self.path, group, "labels", '*.mid')))
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

    def __init__(self, dataset_root_dir=".", path='data/synthesize_trumpet', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', TOTAL_DEBUG=False, logdir="runs/logdir"):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'all'], sequence_length, seed, refresh, device, TOTAL_DEBUG, logdir)

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        # flacs = sorted(glob(os.path.join(self.path, "audio", '*solo*.wav')))
        # midis = sorted(glob(os.path.join(self.path, "labels", '*solo*.jams.mid')))
        if group in self.available_groups():
            flacs = sorted(
                glob(os.path.join(self.path, group, "audio", '*.wav')))
            midis = sorted(
                glob(os.path.join(self.path, group, "labels", '*.mid')))
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

def find_unique_elements_for_lists(list1, list2):
    list1_unique_elements = []
    list2_unique_elements = []
    for element in list1:
        if element not in list2:
            list1_unique_elements.append(element)
    for element in list2:
        if element not in list1:
            list2_unique_elements.append(element)
    return list1_unique_elements, list2_unique_elements

class SynthesizedInstruments(PianoRollAudioDataset):

    def __init__(self, dataset_root_dir=".",  path='data/synthesize', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', TOTAL_DEBUG=False, logdir="runs/logdir"):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'all'], sequence_length, seed, refresh, device, TOTAL_DEBUG, logdir)

    @classmethod
    def available_groups(cls):
        return ['train', 'test', 'val']

    def files(self, group):
        # flacs = sorted(glob(os.path.join(self.path, "audio", '*solo*.wav')))
        # midis = sorted(glob(os.path.join(self.path, "labels", '*solo*.jams.mid')))
        flacs = []
        midis = []
        if group in self.available_groups():
            for paths in glob(self.path+"*"):
                print(f"Adding instrument from {paths} to {group}")
                flacs += sorted(
                    glob(os.path.join(paths, group, "audio", '*.wav')))
                midis += sorted(
                    glob(os.path.join(paths, group, "labels", '*.mid')))
                midis.extend(sorted(
                    glob(os.path.join(paths, group, "labels", '*.tsv'))))
                if len(flacs) != len(midis):

                    set_flacs_without_extension = sorted([element.split("/")[-1].split(".")[0].replace("_hex", "") for element in flacs])
                    set_midis_without_extension = sorted([element.split("/")[-1].split(".")[0] for element in midis])
                    print(f"###### DEBUG: audio files list: {set_flacs_without_extension} \n")
                    print(f"###### DEBUG: labels list: {set_midis_without_extension} \n")
                    print(f"###### DEBUG: lists sizes: audio: {len(set_flacs_without_extension)}, label: {len(set_midis_without_extension)} \n")
                    audio_unique, label_unique = find_unique_elements_for_lists(set_flacs_without_extension, set_midis_without_extension)
                    print(f"###### DEBUG: audio unique size {len(audio_unique)} label unique size: {len(label_unique)}")
                    print(f"###### DEBUG: audio unique: {audio_unique} \n")
                    print(f"###### DEBUG: label unique: {label_unique} \n")
                    raise RuntimeError(f'Detected {len(midis)} labels for {len(flacs)} audio files!')
            files = list(zip(flacs, midis))
            print(f"Number of audio samples: {len(flacs)}, Number of labels: {len(midis)}")
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')

        result = []
        for audio_path, midi_path in files:
            if(os.path.exists(midi_path) and midi_path[-4:] == ".tsv"):
                result.append((audio_path, midi_path))
            else: #if element is mid
                tsv_filename = midi_path[-4]+".tsv"
                if not os.path.exists(tsv_filename):
                    midi = parse_midi(midi_path)
                    np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t',
                            header='onset,offset,note,velocity')
                    os.remove(midi_path)
                result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, dataset_root_dir=".",  path='data/MAPS', groups=None, sequence_length=None, overlap=True, seed=42, refresh=False, device='cpu', TOTAL_DEBUG=False, logdir="runs/logdir"):
        self.overlap = overlap
        print("Initializing MAPS dataset!")
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, refresh, device, TOTAL_DEBUG, logdir)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
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
    def __init__(self, dataset_root_dir=".",  path='MAPS', groups=None, sequence_length=None, overlap=True, seed=42, refresh=False, device='cpu', TOTAL_DEBUG=False, logdir="runs/logdir"):
        self.overlap = overlap
        print(f"Initializing original MAPS dataset with {groups}!")
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, refresh, device, TOTAL_DEBUG=TOTAL_DEBUG, logdir=logdir)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        if group in self.available_groups():
            path_to_wavs_list = os.path.join(self.path, f"list_of_maps_wavs_{group}.np")
            if(not os.path.exists(path_to_wavs_list) or self.refresh == True):
                list_of_wavs = glob(self.path+"/**/"+ f'*{group}.wav', recursive=True)
                list_of_wavs = list(filter(lambda elem: "MUS" in elem, list_of_wavs))
                np.save(path_to_wavs_list, list_of_wavs)
            else:
                print(f"Loading existing wav filelist for {group} group!")
                list_of_wavs = np.load(path_to_wavs_list)
            path_to_midis_list = os.path.join(self.path, f"list_of_maps_midis_{group}.np")
            if(not os.path.exists(path_to_midis_list) or self.refresh == True):
                list_of_midis = glob(self.path+"/**/"+ f'*{group}.mid', recursive=True)
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

class MusicNet(PianoRollAudioDataset):
    def __init__(self, dataset_root_dir=".", path='MusicNet/', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu', TOTAL_DEBUG=False, logdir="runs/logdir"):
        super().__init__(path, dataset_root_dir, groups if groups is not None else [
            'train'], sequence_length, seed, refresh, device, TOTAL_DEBUG, logdir)

    @classmethod
    def available_groups(cls):
        return ['train', 'test']

    def files(self, group):
        if group == 'small test':
            types = ('2303.flac', '2382.flac', '1819.flac')
            flacs = []
            for i in types:
                flacs.extend(glob(os.path.join(self.path, 'test_data', i)))
            flacs = sorted(flacs)
            tsvs = sorted(
                glob(os.path.join(self.path, f'tsv_test_labels/*.tsv')))
        else:
            flacs = sorted(
                glob(os.path.join(self.path, f'{group}_data/*.flac')))
            tsvs = sorted(
                glob(os.path.join(self.path, f'tsv_{group}_labels/*.tsv')))

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return zip(flacs, tsvs)
