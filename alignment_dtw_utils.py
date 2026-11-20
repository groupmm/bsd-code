import os

import librosa
import scipy
import numpy as np
import pandas as pd

from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw, sync_via_mrmsdtw_with_anchors
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic
from synctoolbox.feature.csv_tools import read_csv_to_df, df_to_pitch_features, df_to_pitch_onset_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

from tqdm import tqdm
from glob import glob

import pyloudnorm as pln
import soundfile as sf
from libsoni import sonify_pianoroll_clicks


def get_features_from_audio(audio, tuning_offset, Fs, feature_rate, visualize=False):
    
    f_pitch = audio_to_pitch_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, feature_rate=feature_rate, verbose=visualize)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    
    f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=visualize)
    f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset, feature_rate=feature_rate, feature_sequence_length=f_chroma_quantized.shape[1], visualize=visualize)
    
    return f_chroma_quantized, f_DLNCO

def get_features_from_annotation(df_annotation, feature_rate, visualize=False):
    
    f_pitch = df_to_pitch_features(df_annotation, feature_rate=feature_rate)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch) 
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
       
    f_pitch_onset = df_to_pitch_onset_features(df_annotation)
    f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset,
                                            feature_rate=feature_rate,
                                            feature_sequence_length=f_chroma_quantized.shape[1]+1,
                                            visualize=visualize)
    
    # mute last frame to loosen the end matching restriction of DTW
    f_chroma_quantized[:, -2:] = 0
    f_DLNCO[:, -2:] = 0

    return f_chroma_quantized, f_DLNCO[:, :-1]


def pitch_onset_matrix_to_DLNCO(pitch_onset_matrix : np.ndarray,
                                  feature_sequence_length: int,
                                  feature_rate: int = 50,
                                  midi_min: int = 21,
                                  midi_max: int = 108,
                                  log_compression_gamma: float = 10000.0,
                                  chroma_norm_ord: int = 2,
                                  LN_maxfilterlength_seconds: float = 0.8,
                                  LN_maxfilterthresh: float = 0.1,
                                  DLNCO_filtercoef: np.ndarray = np.sqrt(1 / np.arange(1, 11)),
                                  visualize=False) -> np.ndarray:
    """Computes decaying locally adaptive normalized chroma onset (DLNCO) features from
    a dictionary of peaks obtained e.g. by ``audio_to_pitch_onset_features``.

    Parameters
    ----------
    f_peaks : dict
        A dictionary of onset peaks

            * Each key corresponds to the midi pitch number

            * Each value f_peaks[midi_pitch] is an array of doubles of size 2xN:

                + First row give the positions of the peaks in milliseconds.

                + Second row contains the corresponding magnitudes of the peaks.

    feature_sequence_length : int
        Desired length of the resulting feature sequence. This should be at least as long as the
        position of the last peak in ``f_peaks``, but can be longer.

    feature_rate : int
        Desired features per second in the output representation

    midi_min : int
        Minimum MIDI pitch index (default: 21)

    midi_max : int
        Maximum MIDI pitch index (default: 108)

    log_compression_gamma : float
        Gamma factor of the log compression applied to peak magnitudes.
        
    chroma_norm_ord : int
        Order of the norm used for chroma onset vectors.

    LN_maxfilterlength_seconds : float
        Length of the maximum filter applied for determining local norm of chroma onsets in seconds.

    LN_maxfilterthresh : float
        Minimum threshold for normalizing chroma onsets using local norm.

    DLNCO_filtercoef : np.ndarray
        Sequence of decay coefficients applied on normalized chroma onsets.

    visualize : bool
        Set `True` to visualize chroma onset features (Default: False)

    Returns
    -------
    f_DLNCO : np.array [shape=(d_dlnco, N_dlnco)]
        Decaying Locally adaptively Normalized Chroma Onset features
    """
    
    f_CO = pitch_to_chroma(pitch_onset_matrix).T

    # No two ways to normalize F_CO: simply columnwise (f_N) or via local
    # normalizing curve (f_LN)
    f_N = np.zeros(feature_sequence_length)

    for k in range(feature_sequence_length):
        f_N[k] = np.linalg.norm(f_CO[k, :], chroma_norm_ord)

    f_LN = np.array(f_N, copy=True)
    f_left = np.array(f_N, copy=True)
    f_right = np.array(f_N, copy=True)
    LN_maxfilterlength_frames = int(LN_maxfilterlength_seconds * feature_rate)
    if LN_maxfilterlength_frames % 2 == 1:
        LN_maxfilterlength_frames -= 1
    shift = int(np.floor((LN_maxfilterlength_frames) / 2))

    for s in range(shift):
        f_left = np.roll(f_left, 1, axis=0)
        f_left[0] = 0
        f_right = np.roll(f_right, -1, axis=0)
        f_right[-1] = 0
        f_LN = np.max([f_left, f_LN, f_right], axis=0)

    f_LN = np.maximum(f_LN, LN_maxfilterthresh)

    # Compute f_NC0 (normalizing f_C0 using f_N)
    # f_NCO = np.zeros((feature_sequence_length, 12))

    # Compute f_LNC0 (normalizing f_C0 using f_LN)
    f_LNCO = np.zeros((feature_sequence_length, 12))
    for k in range(feature_sequence_length):
        # f_NCO[k, :] = f_CO[k, :] / (f_N[k]) #+ eps)
        f_LNCO[k, :] = f_CO[k, :] / f_LN[k]

    # Compute f_DLNCO
    f_DLNCO = np.zeros((feature_sequence_length, 12))

    num_coef = DLNCO_filtercoef.size
    for p_idx in range(12):
        v_shift = np.array(f_LNCO[:, p_idx], copy=True)
        v_help = np.zeros((feature_sequence_length, num_coef))

        for n in range(num_coef):
            v_help[:, n] = DLNCO_filtercoef[n] * v_shift
            v_shift = np.roll(v_shift, 1)
            v_shift[0] = 0

        f_DLNCO[:, p_idx] = np.max(v_help, axis=1)

    f_DLNCO = f_DLNCO.T

    return f_DLNCO

def get_features_from_transcription(transcription_features,
                                          audio_length_seconds,
                                          min_midi = 21,
                                          max_midi = 108,
                                          feature_rate = 50):
    onsetsIn = transcription_features["onset_pred"].T
    framesIn = transcription_features["frame_pred"].T
    frame_rate = transcription_features["sample_rate"] / transcription_features["hop_length"]
    idx_interp = np.floor(np.arange(0, audio_length_seconds, 1/feature_rate)*frame_rate).astype(int)

    onsetsIn_res = onsetsIn[:,idx_interp]
    framesIn_res = framesIn[:,idx_interp]

    f_frames_transcription = np.zeros((128, framesIn_res.shape[1]))
    f_frames_transcription[min_midi:max_midi+1,:] = framesIn_res

    f_onsets_transcription = np.zeros((128, onsetsIn_res.shape[1]))
    f_onsets_transcription[min_midi:max_midi+1,:] = onsetsIn_res


    f_chroma_transcriptions = pitch_to_chroma(f_pitch=f_frames_transcription)
    f_chroma_quantized_transcription = quantize_chroma(f_chroma = f_chroma_transcriptions)


    f_DLNCO_transcription = pitch_onset_matrix_to_DLNCO(pitch_onset_matrix = f_onsets_transcription, feature_rate=feature_rate, 
                                                            feature_sequence_length=f_chroma_quantized_transcription.shape[1], visualize=False)
    
    return f_chroma_quantized_transcription, f_DLNCO_transcription
    

def sonify(audio, df, sr, out_path = None, target_loudness=-25):
    df = df.copy()  # Avoid modifying the original DataFrame
    df["label"] = "" # libsoni wants a label column
    if "start_sec" in df.columns and "duration_sec" in df.columns:
        df["start"] = df["start_sec"]
        df["duration"] = df["duration_sec"]
    meter = pln.Meter(rate = sr)
    audio_loudness = meter.integrated_loudness(audio)
    audio = pln.normalize.loudness(audio, audio_loudness, target_loudness)

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

    return stereo_audio_1