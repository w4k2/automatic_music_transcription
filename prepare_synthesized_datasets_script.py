from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from IPython.utils import io
import shutil
import pretty_midi
import pprint
import os
import sys
from midi2audio import FluidSynth
from tsv_label import TsvLabel
fs = FluidSynth()

pp = pprint.PrettyPrinter(indent=4)


def get_all_guitarset_midis(directory):
    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for file in filenames:
            if file[-3:] == "mid":
                listOfFiles.append(os.path.join(dirpath, file))
    return listOfFiles


def get_all_MAPS_tsvs(MAPS_subdirectory, directory):
    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for file in filenames:
            file_expression = MAPS_subdirectory+".tsv"
            if file_expression in file:
                listOfFiles.append(os.path.join(dirpath, file))
    return listOfFiles


def convert_to_another_instrument(midi, program=1):
    new_instruments_list = []
    for instrument in midi.instruments:
        new_instruments_list.append(pretty_midi.Instrument(program=program))
        for note in instrument.notes:
            new_instruments_list[-1].notes.append(note)
    midi.instruments = new_instruments_list
    return midi

# MODIFY THIS LIST TO GENERATE ADDITIONAL SYNTHESIZED INSTRUMENTS
instrument_list = [
    # 'Bright Acoustic Piano',
    # 'Electric Grand Piano',
    # 'Honky-tonk Piano',
    # 'Electric Piano 1',
    # 'Electric Piano 2',
    # 'Harpsichord',
    # 'Clavinet',
    # 'Celesta',
    # 'Glockenspiel',
    #'Music Box',
    # 'Vibraphone',
    # 'Marimba',
    # 'Xylophone',
    #'Tubular Bells',
    # 'Dulcimer',
    #'Drawbar Organ',
    #'Percussive Organ',
    #'Rock Organ',
    #'Church Organ',
    #'Reed Organ',
    # 'Accordion',
    # 'Harmonica',
    #'Tango Accordion',
    #'Acoustic Guitar (nylon)',
    'Acoustic Guitar (steel)',
    #'Electric Guitar (jazz)',
    #'Electric Guitar (clean)',
    #'Electric Guitar (muted)',
    #'Overdriven Guitar',
    #'Distortion Guitar',
    #'Guitar Harmonics',
    #'Acoustic Bass',
    #'Electric Bass (finger)',
    #'Electric Bass (pick)',
    #'Fretless Bass',
    #'Slap Bass 1',
    #'Slap Bass 2',
    #'Synth Bass 1',
    #'Synth Bass 2',
    #'Violin',
    # 'Viola',
    # 'Cello',
    # 'Contrabass',
    #'Tremolo Strings',
    #'Pizzicato Strings',
    #'Orchestral Harp',
    # 'Timpani',
    #'String Ensemble 1',
    #'String Ensemble 2',
    #'Synth Strings 1',
    #'Synth Strings 2',
    #'Choir Aahs',
    #'Voice Oohs',
    #'Synth Choir',
    #'Orchestra Hit',
    # 'Trumpet',
    # 'Trombone',
    # 'Tuba',
    #'Muted Trumpet',
    #'French Horn',
    #'Brass Section',
    #'Synth Brass 1',
    #'Synth Brass 2',
    #'Soprano Sax',
    #'Alto Sax',
    #'Tenor Sax',
    #'Baritone Sax',
    # 'Oboe',
    #'English Horn',
    # 'Bassoon',
    # 'Clarinet',
    # 'Piccolo',
    #'Flute',
    # 'Recorder',
    #'Pan Flute',
    #'Blown bottle',
    # 'Shakuhachi',
    # 'Whistle',
    # 'Ocarina',
    'Acoustic Grand Piano'
]

print("Generating datasets for following instruments:")
print(instrument_list)


def standarize_name(name):
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '')


def make_standard_directories(path):
    print(f"Creating directory {path} test/val/train...")
    os.makedirs(f'{path}')
    os.makedirs(f'{path}/test')
    os.makedirs(f'{path}/val')
    os.makedirs(f'{path}/train')


def copy_labels_to_target_dir(label_list, target_dir):
    print(f"Copying labels to {target_dir}/labels...")
    os.makedirs(f'{target_dir}/labels')
    for label in label_list:
        shutil.copy(label, target_dir+"/labels/")

def remove_if_exists(filename):
    if os.path.exists(filename):
        print(f"Removing {filename} file!")
        os.remove(filename)
    else:
        print(f"Can not delete {filename} as it doesn't exists")

def synthesize_instrument_for_labels(labels, output_directory, instrument_name):

    program = pretty_midi.instrument_name_to_program(instrument_name)
    print(
        f"Detected program for given instrument {instrument_name}: {program} - copying to {output_directory} directory!")
    os.makedirs(f'{output_directory}')
    for label in labels:
        if label[-3:] == "mid":
            try:
                midi = pretty_midi.PrettyMIDI(label)
                midi = convert_to_another_instrument(midi, program)
                midi.write(
                    f'{output_directory}/{os.path.basename(label)}_temp.mid')
                with io.capture_output() as captured:
                    fs.midi_to_audio(f'{output_directory}/{os.path.basename(label)}_temp.mid',
                                    f'{output_directory}/{os.path.basename(label)}.wav')
                os.remove(f'{output_directory}/{os.path.basename(label)}_temp.mid')
                os.rename(f'{output_directory}/{os.path.basename(label)}.wav',
                        f'{output_directory}/{os.path.basename(label)[:-9]}_hex.wav')
            except:
                print(f"File {label} generation for {instrument_name} needs to be skipped - problem detected during midi analysis")
                remove_if_exists(f'{output_directory}/{os.path.basename(label)}_temp.mid')
                remove_if_exists(f'{output_directory}/{os.path.basename(label)[:-9]}_hex.wav')
                remove_if_exists(f'{output_directory}/{os.path.basename(label)}.wav')
        elif label[-3:] == 'tsv':
            try:
                tsv_label = TsvLabel(label)
                tsv_midi = tsv_label.to_pretty_midi()
                tsv_midi = convert_to_another_instrument(tsv_midi, program)
                tsv_midi.write(
                    f'{output_directory}/{os.path.basename(label)}_temp.mid')
                with io.capture_output() as captured:
                    fs.midi_to_audio(f'{output_directory}/{os.path.basename(label)}_temp.mid',
                                 f'{output_directory}/{os.path.basename(label)}.wav')
                os.remove(f'{output_directory}/{os.path.basename(label)}_temp.mid')
            except Exception as e:
                print(f"File {label} generation for {instrument_name} needs to be skipped - problem detected during tsv analysis")
                remove_if_exists(f'{output_directory}/{os.path.basename(label)}_temp.mid')
                remove_if_exists(f'{output_directory}/{os.path.basename(label)}.wav')
        else:
            raise Exception(f"BUG! UNSUPPORTED FILE - {label}")


def create_files_for_phase(directory, labels, instrument, phase):
    directory_name = f'{GLOBAL_PATH}/data/synthesized_{standarize_name(instrument)}'
    os.makedirs(os.path.join(directory, phase))
    copy_labels_to_target_dir(labels, f'{directory_name}/{phase}')
    synthesize_instrument_for_labels(labels,
                                     f'{directory_name}/{phase}/audio',
                                     instrument)


def generate_datasets_for_instruments(instrument):
    # GuitarSet part of synthesized dataset
    train_filenames = []
    val_filenames = []
    test_filenames =[]
    if guitarset:
        print(f"Aquiring guitarset data for {instrument} instrument!")
        train_filenames = get_all_guitarset_midis("data/guitarset/train")
        val_filenames = get_all_guitarset_midis("data/guitarset/val")
        test_filenames = get_all_guitarset_midis("data/guitarset/test")
    if maps:
        print(f"Aquiring MAPS data for {instrument} instrument!")
        # MAPS part of synthesized dataset
        for group in ('AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl',
                        'StbgTGd2'):
            train_filenames.extend(get_all_MAPS_tsvs(
                group, "data/MAPS/tsv/matched"))

        val_filenames.extend(get_all_MAPS_tsvs(
            'ENSTDkAm', "data/MAPS/tsv/matched"))
        test_filenames.extend(get_all_MAPS_tsvs(
            'ENSTDkCl', "data/MAPS/tsv/matched"))

    directory_name = f'{GLOBAL_PATH}/data/synthesized_{standarize_name(instrument)}'
    os.makedirs(directory_name)
    create_files_for_phase(
        directory_name, train_filenames, instrument, 'train')
    create_files_for_phase(directory_name, test_filenames, instrument, 'test')
    create_files_for_phase(directory_name, val_filenames, instrument, 'val')

# generate_datasets_for_instruments(instrument_list)


assert (len(sys.argv) == 2)
global GLOBAL_PATH
global guitarset
global maps

GLOBAL_PATH = sys.argv[1]
guitarset = False
maps = True
with Pool(2) as pool:
    os.mkdir(GLOBAL_PATH)
    pool.map(generate_datasets_for_instruments, instrument_list)
