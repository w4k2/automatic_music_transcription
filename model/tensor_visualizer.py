import torch
import os
from PIL import Image
import soundfile as sf
from .decoding import extract_notes_wo_velocity
from .midi import save_midi_new_version


class TensorVisualizer:
    def __init__(self, logdir):
        self.dictionary_of_existing_filenames = {}
        self.logdir = logdir

    def save_pianoroll_from_pytorch_tensor(self, tensor, audio_path, index):
        onsets = (1 - (tensor.t() > 0.5).to(torch.uint8)).cpu()
        frames = (1 - (tensor.t() > 0.5).to(torch.uint8)).cpu()
        both = (1 - (1 - onsets) * (1 - frames))
        image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
        image = Image.fromarray(image, 'RGB')
        image = image.resize((image.size[0], image.size[1] * 4))
        
        
        destination_filename = os.path.basename(audio_path)+f"_{index}.png"
        if destination_filename in self.dictionary_of_existing_filenames.keys():
            self.dictionary_of_existing_filenames[destination_filename] += 1
        else:
            self.dictionary_of_existing_filenames[destination_filename] = 0
        image_save_directory = os.path.join(self.logdir, "ground_truth")
        if not os.path.exists(image_save_directory):
            os.makedirs(image_save_directory)
        epoch_directory = os.path.join(image_save_directory, f"{self.dictionary_of_existing_filenames[destination_filename]}")
        if not os.path.exists(epoch_directory):
            os.makedirs(epoch_directory)
        image_save_filename = os.path.join(epoch_directory, destination_filename)
        image.save(image_save_filename)
        print(f"BLAX pianoroll path: {image_save_filename}, shape: {tensor.shape}, index: {index}")

    def save_audio_from_pytorch_tensor(self, tensor, audio_path, index):
        audio = tensor.detach().cpu().numpy()
        audio_save_directory = os.path.join(self.logdir, "recordings")
        if not os.path.exists(audio_save_directory):
            os.makedirs(audio_save_directory)
        destination_filename = os.path.basename(audio_path)+f"_audio_{index}.wav"
        if destination_filename in self.dictionary_of_existing_filenames.keys():
            self.dictionary_of_existing_filenames[destination_filename] += 1
        else:
            self.dictionary_of_existing_filenames[destination_filename] = 0
        epoch_directory = os.path.join(audio_save_directory, f"{self.dictionary_of_existing_filenames[destination_filename]}")
        if not os.path.exists(epoch_directory):
            os.makedirs(epoch_directory)
        audio_save_filename = os.path.join(epoch_directory, destination_filename)
        sf.write(audio_save_filename, audio, 16000, subtype='PCM_U8')
        print(f"BLAX audio path: {audio_path}, shape: {tensor.shape}, index: {index}")

    def save_image_from_pytorch_tensor(self, tensor, audio_path, index, optional_directory=None):
        onsets = (1 - (tensor.t() > 0.5).to(torch.uint8)).cpu()
        frames = (1 - (tensor.t() > 0.5).to(torch.uint8)).cpu()
        both = (1 - (1 - onsets) * (1 - frames))
        image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
        image = Image.fromarray(image, 'RGB')
        image = image.resize((image.size[0], image.size[1] * 4))
        
        if(os.path.basename(audio_path) != ""):
            destination_filename = os.path.basename(audio_path)+f".png"
        else:
            destination_filename = f"{index}_label.png"
        image_save_directory = os.path.join(self.logdir, "general_images")
        if not os.path.exists(image_save_directory):
            os.makedirs(image_save_directory)
        if optional_directory != None:
            image_save_directory = os.path.join(image_save_directory, optional_directory)
            if not os.path.exists(image_save_directory):
                os.makedirs(image_save_directory)
        batch_directory = os.path.join(image_save_directory, str(index))
        if not os.path.exists(batch_directory):
            os.makedirs(batch_directory)
        image_save_filename = os.path.join(batch_directory, destination_filename)
        image.save(image_save_filename)
        # print(f"BLAX pianoroll path: {image_save_filename}, shape: {tensor.shape}, index: {index}")
    
    def save_general_audio_from_pytorch_tensor(self, tensor, audio_path, index, optional_directory=None):
        audio = tensor.detach().cpu().numpy()
        audio_save_directory = os.path.join(self.logdir, "recordings")
        if not os.path.exists(audio_save_directory):
            os.makedirs(audio_save_directory)
        if(os.path.basename(audio_path) != ""):
            destination_filename = os.path.basename(audio_path)+f"_audio.wav"
        else:
            destination_filename = f"{index}_audio.wav"
        
        image_save_directory = os.path.join(self.logdir, "general_images")
        if not os.path.exists(image_save_directory):
            os.makedirs(image_save_directory)
        if optional_directory != None:
            image_save_directory = os.path.join(image_save_directory, optional_directory)
            if not os.path.exists(image_save_directory):
                os.makedirs(image_save_directory)
        batch_directory = os.path.join(image_save_directory, str(index))
        if not os.path.exists(batch_directory):
            os.makedirs(batch_directory)
        audio_save_filename = os.path.join(batch_directory, destination_filename)
        sf.write(audio_save_filename, audio, 16000, subtype='PCM_U8')

    def save_general_midi_from_pytorch_tensor(self, pianoroll, audio_path, index, optional_directory=None, pianoroll_description="pianoroll"):
        p_est, i_est = extract_notes_wo_velocity(pianoroll, pianoroll, 0.5, 0.5)
        audio_save_directory = os.path.join(self.logdir, "recordings")
        if not os.path.exists(audio_save_directory):
            os.makedirs(audio_save_directory)
        
        if(os.path.basename(audio_path) != ""):
            destination_filename = os.path.basename(audio_path)+f"_{pianoroll_description}.mid"
        else:
            destination_filename = f"{index}_{pianoroll_description}.mid"
        image_save_directory = os.path.join(self.logdir, "general_images")
        if not os.path.exists(image_save_directory):
            os.makedirs(image_save_directory)
        if optional_directory != None:
            image_save_directory = os.path.join(image_save_directory, optional_directory)
            if not os.path.exists(image_save_directory):
                os.makedirs(image_save_directory)
        batch_directory = os.path.join(image_save_directory, str(index))
        if not os.path.exists(batch_directory):
            os.makedirs(batch_directory)
        midi_filename = os.path.join(batch_directory, destination_filename)
        try:
            save_midi_new_version(midi_filename, p_est, i_est, [127]*len(p_est))
        except:
            print("BUG CATCHED")
            print(midi_filename)
            print(p_est)
            print(i_est)
            save_midi_new_version(midi_filename, p_est, i_est, [127]*len(p_est), debug="on")
            assert(False)
