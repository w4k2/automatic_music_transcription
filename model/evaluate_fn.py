import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
import mir_eval
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, average_precision_score, f1_score
from scipy.stats import hmean
from tqdm import tqdm

from model import *
from model.instrument_recognition_model import InstrumentRecognitionModel
from Levenshtein import distance
from torch.utils.data import DataLoader


eps = sys.float_info.epsilon


def evaluate_classification(data_loader, model: InstrumentRecognitionModel, batch_size, save_path=None, classes=None):
    metrics = defaultdict(list)
    metrics_functions = [("balanced_accuracy_score", balanced_accuracy_score),
                         ("average_precision_score", average_precision_score),
                         ("precision_score", precision_score),
                         ("recall_score", recall_score),
                         ("f1_score", f1_score)]
    result_dict = {}
    softmax = nn.Softmax(dim=1)
    for batch in data_loader:
        #print(f"Debug - batch {batch}")
        pred, losses, _ = model.run_on_batch(batch)
        for key, loss in losses.items():
            metrics[key].append(loss.item())
        for key, value in pred.items():
            value.squeeze_(0).relu_()
        # print(softmax(pred['results']))
        #print(f"Debug - predictions {pred['results']}")
        predictions = (softmax(pred['results']) > 0.5).long()
        if classes != None:
            print("detailed predictions: ", [classes[elem] for elem in torch.argmax(
                predictions, dim=1).to('cpu').numpy().tolist()])
            print("detailed label: ", [classes[elem] for elem in torch.argmax(
                batch['label'], dim=1).to('cpu').numpy().tolist()])
        else:
            print("predictions:", torch.argmax(predictions, dim=1))
            print("label: ", torch.argmax(batch['label'], dim=1))
        pred = predictions[0].cpu()
        target = batch['label'][0].cpu()
        for metric in metrics_functions:
            if metric[0] in result_dict:
                result_dict[metric[0]] += metric[1](target, pred)
            else:
                result_dict[metric[0]] = metric[1](target, pred)

    for key in result_dict.keys():
        result_dict[key] = result_dict[key] / len(data_loader)
        metrics[f'metrics/overall/{key}'] = result_dict[key]

    return metrics


def evaluate_wo_velocity(data, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    metrics = defaultdict(list)

    for label in data:
        pred, losses, _ = model.run_on_batch(label, "evaluation")

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            value.squeeze_(0).relu_()

        p_ref, i_ref = extract_notes_wo_velocity(
            label['onset'], label['frame'])
        p_est, i_est = extract_notes_wo_velocity(
            pred['onset'], pred['frame'], onset_threshold, frame_threshold)

        predictions = (pred['onset'] > 0.5).long()
        labels = (label['onset'] > 0.5).long()
        levensthein_distance = 0
        for i in range(predictions.shape[0]):
            levensthein_distance += distance(
                str(predictions[i].tolist()), str(labels[i].tolist()))
        metrics['metric/note/levensthein'].append(levensthein_distance)

        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)
        t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        # Converting time steps to seconds and midi number to frequency
        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi)
                          for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi)
                          for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(
            i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(
            hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        avp = average_precision_score(label['frame'].cpu().detach(
        ).flatten(), pred['frame'].cpu().detach().flatten())
        metrics['metric/MusicNet/micro_avg_P'].append(avp)

        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')
                    ].append(loss)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            label_path = os.path.join(
                save_path, os.path.basename(label['path']) + '.label.png')
            save_pianoroll(label_path, label['onset'], label['frame'])
            pred_path = os.path.join(
                save_path, os.path.basename(label['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['onset'], pred['frame'])
            midi_path = os.path.join(
                save_path, os.path.basename(label['path']) + '.pred.mid')
            save_midi(midi_path, p_est, i_est, [127]*len(p_est))

    return metrics
