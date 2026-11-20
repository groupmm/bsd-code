import os
import pretty_midi

from libsoni import sonify_pianoroll_clicks
import pyloudnorm as pln
import soundfile as sf
import numpy as np

import numpy as np
import pandas as pd
from librosa import midi_to_hz
import pretty_midi

import torch
from collections import defaultdict

def get_peaks(notes, win_size):
    constraints = []
    for i in range(1, win_size + 1):
        forward = torch.roll(notes, i, 0)
        forward[: i, ...] = 0  # assume time axis is 0
        backward = torch.roll(notes, -i, 0)
        backward[-i:, ...] = 0
        constraints.extend([forward, backward])
    res = torch.ones(notes.shape, dtype=bool)
    for elem in constraints:
        res = res & (notes >= elem)
    return res


def sonify(audio, df, sr, out_path=None, target_loudness=-25):
    df["label"] = "" # libsoni wants a label column
    meter = pln.Meter(rate = sr)
    audio_loudness = meter.integrated_loudness(audio)
    audio = pln.normalize.loudness(audio, audio_loudness, target_loudness-2)

    x_peaks = sonify_pianoroll_clicks(pianoroll_df=df[['start', 'duration', 'pitch', 'velocity', 'label']], fs=sr)
    x_peaks = pln.normalize.loudness(x_peaks, meter.integrated_loudness(x_peaks), target_loudness)
    shape_diff = len(x_peaks)-len(audio)
    if shape_diff>0:
        audio = np.concatenate((audio, np.zeros(shape_diff)))
    elif shape_diff<0:
        x_peaks = np.concatenate((x_peaks, np.zeros(abs(shape_diff))))

    stereo_audio_1 = np.array([audio, x_peaks]).T  # Combine and transpose for stereo
    
    if out_path is not None:
        sf.write(out_path, samplerate=sr, data=stereo_audio_1)
    df.drop(columns=["label"], inplace=True) # Clean up df
    return stereo_audio_1
