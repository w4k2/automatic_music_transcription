import os
import sys
import librosa
import torch
from model.instrument_recognition_model import InstrumentRecognitionModel
from model.utils import summary
from model.constants import *
from classesstxt_utils import load_classess

# parameters for the network
ds_ksize, ds_stride = (2, 2), (2, 2)
mode = 'imagewise'
sparsity = 1

if __name__ == '__main__':
    audio_path = sys.argv[1]
    GPU = '0'
    device = 'cuda'
    resume_iteration = "150"
    spec = 'CQT'
    pretrained_model_path = "runs/InstrumentClassification-CQT-imagewise_norm-Transcriber_only-230115-120819"
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    
    classes = load_classess(pretrained_model_path)

    # Assume that the checkpoint is in this folder
    trained_dir = pretrained_model_path
    model = InstrumentRecognitionModel(ds_ksize, ds_stride, mode=mode,
                                    spec=spec, norm=sparsity, device=device, number_of_instruments=len(classes))
    model.to(device)
    model_path = os.path.join(pretrained_model_path, f'model-{resume_iteration}.pt')
    model_state_dict = torch.load(
        model_path, map_location=torch.device(device))
    model.load_my_state_dict(model_state_dict)

    summary(model)
    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        audio, sr = librosa.load(
                audio_path, dtype='float32', mono=True, sr=SAMPLE_RATE)
        assert(sr == SAMPLE_RATE)
        audio_label = torch.FloatTensor(audio).to(device)
        spec = model.spectrogram(audio_label)
        spec = torch.log(spec + 1e-5)
        spec = model.normalize.transform(spec)
        spec = spec.transpose(-1, -2)
        classification_results = model(
            spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))
        print(f"Debug - audio {audio_label}")
        print(f"Debug - predictions {classification_results}")
        predictions = (softmax(classification_results) > 0.5).long()
        print("prediction: ", [classes[elem] for elem in torch.argmax(predictions, dim=1).to('cpu').numpy().tolist()])
