import os
import glob

# torch.set_printoptions(profile="full")
def extract_opus(filename):
    opus = os.path.basename(filename).split(".")[0]
    if "_hex" in opus:
        opus=opus[:-4]
    return opus

def find_label_for_given_wav(path, wav_filename):
    opus = extract_opus(wav_filename)
    list_of_labels = glob(f"{path}**/*{opus}*")
    if len(list_of_labels):
        return list_of_labels[0]
    else:
        return None

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

def check_consistency_of_size_of_audio_and_labels(wavs, labels):
    if len(wavs) != len(labels):
        set_flacs_without_extension = sorted([element.split("/")[-1].split(".")[0].replace("_hex", "") for element in wavs])
        set_midis_without_extension = sorted([element.split("/")[-1].split(".")[0] for element in labels])
        print(f"###### DEBUG: audio files list: {set_flacs_without_extension} \n")
        print(f"###### DEBUG: labels list: {set_midis_without_extension} \n")
        print(f"###### DEBUG: lists sizes: audio: {len(set_flacs_without_extension)}, label: {len(set_midis_without_extension)} \n")
        audio_unique, label_unique = find_unique_elements_for_lists(set_flacs_without_extension, set_midis_without_extension)
        print(f"###### DEBUG: audio unique size {len(audio_unique)} label unique size: {len(label_unique)}")
        print(f"###### DEBUG: audio unique: {audio_unique} \n")
        print(f"###### DEBUG: label unique: {label_unique} \n")
        raise RuntimeError(f'Detected {len(labels)} labels for {len(wavs)} audio files!')

def prepare_list_of_tuples_with_audio_and_label_filenames(zipped_file):
        result = []
        for audio_path, midi_path in zipped_file:
            # check if opus for audio is the same as for midi:
            audio_opus = extract_opus(audio_path)
            midi_opus = extract_opus(midi_path)
            assert audio_opus == midi_opus
            result.append((audio_path, midi_path))
        return result