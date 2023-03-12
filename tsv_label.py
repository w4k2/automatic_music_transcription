import os
import numpy as np
import librosa
import pretty_midi
from mir_eval.util import midi_to_hz

from model.constants import *
from model.decoding import extract_notes
from model.midi import save_midi


class TsvLabel():
    def __init__(self, tsv_path):
        self.tsv_representation = np.loadtxt(
            tsv_path, delimiter='\t', skiprows=1)
        n_steps = int(
            (self.tsv_representation[-1][1]*SAMPLE_RATE) // HOP_LENGTH)
        n_keys = MAX_MIDI - MIN_MIDI + 1
        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        for onset, offset, note, vel in self.tsv_representation:
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
        self.onsets = (label == 3).float()
        self.offset = (label == 1).float()
        self.frames = (label > 1).float()
        self.n_steps = n_steps
        self.label = label
        self.velocity = velocity.float()
        self.tsv_path_base = os.path.basename(tsv_path)
    '''
        label: torch.ByteTensor, shape = [num_steps, midi_bins]
        a matrix that contains the onset/offset/frame labels encoded as:
        3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
    '''

    def get_label(self):
        return self.label

    '''
        velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
        a matrix that contains MIDI velocity values at the frame locations
    '''

    def get_velocity(self):
        return self.velocity

    def get_number_of_samples(self):
        return self.n_steps

    def to_pretty_midi(self):
        scaling = HOP_LENGTH / SAMPLE_RATE

        p_ref, i_ref, v_ref = extract_notes(
            self.onsets, self.frames, self.velocity)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
        i_ref = (i_ref * scaling).reshape(-1, 2)

        save_midi(f"{self.tsv_path_base}_tsvtemp.mid", p_ref, i_ref, v_ref)
        pretty_midi_representation = pretty_midi.PrettyMIDI(
            f"{self.tsv_path_base}_tsvtemp.mid")
        os.remove(f"{self.tsv_path_base}_tsvtemp.mid")
        return pretty_midi_representation

    def to_tsv(self):
        return self.tsv_representation
