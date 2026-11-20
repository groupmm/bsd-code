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

from alignment_dtw_utils import *

from tqdm import tqdm
from glob import glob

from musicxml_parser.csv_to_midi import df_to_midi
from paths import *
# some synctoolbox parameters
Fs = 22050
feature_rate = 50
step_weights = np.array([1.5, 1.5, 2.0])
threshold_rec = 10 ** 6

#Path of Audio recordings
audio_dir = os.path.join(bsd_dir, "02_Audio", "wav_44100_440Hz")

#Path of Annotations
score_dir = os.path.join(bsd_dir, "01_UnalignedNoteAnnotations")

#Path for Start-End Annotations
start_end_dir = os.path.join(bsd_dir, "04_Extra", "ann_start_end")

#Path for Alignment output
csv_out_dir = os.path.join(bsd_dir, "03_NoteAnnotations", "Sequence-Alignment")

transcription_dir = os.path.join(bsd_dir, "04_Extra", "transcriptions_MN-Transcriber")
use_transcription_features = True
 
def orch_sync(audio_path, ann_path, start_end_path, save_dir, Fs=Fs, feature_rate=feature_rate, step_weights=step_weights, threshold_rec = threshold_rec):


    #Load audio
    audio_uncut, _ = librosa.load(audio_path, sr=Fs, mono=True)

    #Load annotation
    df_annotation = pd.read_csv(ann_path, sep=";", index_col="index")
    df_annotation.rename(columns={"start_sec": "start", "end_sec": "end", "duration_sec" : "duration"}, inplace=True)

    drum_mask = df_annotation["pitch"] == -1
    df_annotation_alignment = df_annotation.loc[~drum_mask].copy()

    if df_annotation_alignment.empty:
        raise ValueError(f"No pitched notes available for alignment in {audio_path}")

    #Load Start-End Annotation
    start_end_ann = True
    if not os.path.exists(start_end_path):
        #print(f"Warning, no start end annotations for file {audio_path}")
        audio_start = 0
        audio_end = len(audio_uncut)/Fs
        start_end_ann = False
    else:
        df_startEnd = pd.read_csv(start_end_path, sep=",",header=None)
        audio_start = df_startEnd[0][0]
        audio_end = df_startEnd[0][1]
    

    audio = audio_uncut[int(np.floor(audio_start*Fs)):int(np.min([np.ceil(audio_end*Fs),len(audio_uncut)]))]

    #Making the ann start point equal to audio start
    
    ann_start = np.min(df_annotation_alignment["start"])
    if ann_start!=0:
        df_annotation_alignment["end"] -= ann_start
        df_annotation_alignment["start"] -= ann_start
    if not start_end_ann:
        # Adding a little bit offset to loosen the DTW start matching constraints
        df_annotation_alignment["end"] += 0.1
        df_annotation_alignment["start"] += 0.1
        
    #"stretch" the csv-notes to match the length of the audio recording
    df_annotation_alignment["duration"] = df_annotation_alignment["end"] - df_annotation_alignment["start"]
    annotation_dur = np.max(df_annotation_alignment["start"] + df_annotation_alignment["duration"])-np.min(df_annotation_alignment["start"])
    annotation_stretch_fac = (audio_end-audio_start) / annotation_dur

    df_annotation_alignment["start"] *= annotation_stretch_fac
    df_annotation_alignment["duration"] *= annotation_stretch_fac
    df_annotation_alignment["end"] *= annotation_stretch_fac

    tuning_offset = estimate_tuning(audio, Fs)
    #print('Estimated tuning deviation for recording: %d cents' % (tuning_offset))
    
    #Computing quantized chroma and DLNCO features
    if not use_transcription_features:
        f_chroma_quantized_audio, f_DLNCO_audio = get_features_from_audio(audio, tuning_offset, Fs, feature_rate)
        anchor_points = None
    else:
        # Get features from transcription
        transcription_features = np.load(os.path.join(transcription_dir, os.path.basename(audio_path).replace(".wav", ".npz")))
        f_chroma_quantized_audio, f_DLNCO_audio = get_features_from_transcription(transcription_features=transcription_features,
                                                                                  audio_length_seconds=len(audio_uncut)/Fs,
                                                                                    feature_rate=feature_rate)
        f_chroma_quantized_audio = f_chroma_quantized_audio[:, int(audio_start*feature_rate):int(audio_end*feature_rate)]
        f_DLNCO_audio = f_DLNCO_audio[:, int(audio_start*feature_rate):int(audio_end*feature_rate)]
        anchor_points = None
       


    f_chroma_quantized_annotation, f_DLNCO_annotation = get_features_from_annotation(df_annotation_alignment, feature_rate)

    # If we can't be sure whether or not the audio has a transposition then uncomment the following to check
    #Finding optimal shift of chroma vectors
    #f_cens_1hz_audio = quantized_chroma_to_CENS(f_chroma_quantized_audio, 201, 50, feature_rate)[0]
    #f_cens_1hz_annotation = quantized_chroma_to_CENS(f_chroma_quantized_annotation, 201, 50, feature_rate)[0]
    
    # opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_audio, f_cens_1hz_annotation)
    # if opt_chroma_shift != 0:
    #     with open("chroma_shift_nonzero", "a+") as f:
    #         f.write(f"{audio_path}, {opt_chroma_shift}\n")
    #assert opt_chroma_shift == 0, "Found piece with chroma shift"
    #print('Pitch shift between the audio recording and score, determined by DTW:', opt_chroma_shift, 'bins')
    
    # f_chroma_quantized_annotation = shift_chroma_vectors(f_chroma_quantized_annotation, opt_chroma_shift)
    # f_DLNCO_annotation = shift_chroma_vectors(f_DLNCO_annotation, opt_chroma_shift)

    #Performing MrMsDTW
    wp = sync_via_mrmsdtw_with_anchors(f_chroma1=f_chroma_quantized_audio, 
                        f_onset1=f_DLNCO_audio, 
                        f_chroma2=f_chroma_quantized_annotation, 
                        f_onset2=f_DLNCO_annotation, 
                        input_feature_rate=feature_rate, 
                        step_weights=step_weights, 
                        threshold_rec=threshold_rec,
                        anchor_pairs=anchor_points,
                        verbose=False)

    
    wp = make_path_strictly_monotonic(wp)


    #np.save(os.path.join(save_dir,os.path.basename(audio_path).replace(".wav", ".npy")), wp)


    df_annotation_warped = df_annotation.copy(deep=True)
    df_annotation_warped.loc[df_annotation_alignment.index, ['start', 'end']] = scipy.interpolate.interp1d(wp[1] / feature_rate,
                                wp[0] / feature_rate, kind='linear', fill_value="extrapolate")(df_annotation_alignment[['start', 'end']])
    df_annotation_warped.loc[df_annotation_alignment.index, "duration"] = df_annotation_warped.loc[df_annotation_alignment.index, "end"] - df_annotation_warped.loc[df_annotation_alignment.index, "start"]
    df_annotation_warped.loc[df_annotation_alignment.index, 'start'] += audio_start
    df_annotation_warped.loc[df_annotation_alignment.index, 'end'] += audio_start

    if drum_mask.any():
        aligned_measures = (
            df_annotation_warped.loc[~drum_mask, ["start_meas", "start", "end"]]
            .groupby("start_meas", sort=True)
            .first()
            .sort_index()
        )

        if not aligned_measures.empty:
            measure_positions = aligned_measures.index.to_numpy(dtype=float)
            start_lookup = aligned_measures["start"].to_numpy(dtype=float)
            end_lookup = aligned_measures["end"].to_numpy(dtype=float)
            drum_measures = df_annotation_warped.loc[drum_mask, "start_meas"].to_numpy(dtype=float)
            df_annotation_warped.loc[drum_mask, "start"] = np.interp(drum_measures, measure_positions, start_lookup)
            df_annotation_warped.loc[drum_mask, "end"] = np.interp(drum_measures, measure_positions, end_lookup)
            df_annotation_warped.loc[drum_mask, "duration"] = df_annotation_warped.loc[drum_mask, "end"] - df_annotation_warped.loc[drum_mask, "start"]

    
    base_filename = audio_path.split(os.sep)[-1][:-4]
    # for some note the duration can be zero which causes some problems later on, so we set a minimal note length of 50ms here
    df_annotation_warped.loc[df_annotation_warped["duration"] <= 0, "duration"] = 0.05
    df_annotation_warped["duration"] = np.round(df_annotation_warped['duration'], 3)
    df_annotation_warped["end"] = df_annotation_warped["start"] + df_annotation_warped["duration"]

    df_annotation_warped.rename(columns={"start": "start_sec", "end": "end_sec", "duration" : "duration_sec"}, inplace=True)
    df_annotation_warped["start_sec"] = np.round(df_annotation_warped['start_sec'], 3)
    df_annotation_warped["end_sec"] = np.round(df_annotation_warped['end_sec'], 3)
    df_annotation_warped["duration_sec"] = np.round(df_annotation_warped['duration_sec'], 3)
     
    csv_filename = os.path.join(save_dir, f'{base_filename}.csv')
        
    #Save the DataFrame as a CSV file
    df_annotation_warped.to_csv(csv_filename, sep=';', index=True, index_label="index")
    mid = df_to_midi(df_annotation_warped)
    mid.write(csv_filename.replace(".csv", ".mid"))

    return

def main():
    os.makedirs(csv_out_dir, exist_ok=True)
    audio_list = [f for f in sorted(os.listdir(audio_dir)) if "Beethoven" in f and ".wav" in f]

    audio_path_list = [os.path.join(audio_dir ,f) for f in audio_list]
    print(f"Numbers of audio recordings: {len(audio_path_list)}")

    for audio_path in tqdm(audio_path_list):
        override = False
        file_name = os.path.splitext(os.path.basename(audio_path))[0]

        ann_path = os.path.join(score_dir,f"{file_name}.csv")
        start_end_path = os.path.join(start_end_dir,f"{file_name}.csv")

        out_path = os.path.join(csv_out_dir, f"{file_name}.csv")

        if os.path.exists(out_path) and not override:
            print("Already synced", file_name)
        else:
            orch_sync(audio_path, ann_path, start_end_path, csv_out_dir, Fs=Fs, feature_rate=feature_rate, step_weights=step_weights, threshold_rec = threshold_rec)

if __name__ == "__main__":
    main()