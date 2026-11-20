import os
from tqdm import tqdm
from glob import glob

import librosa
import scipy
import numpy as np
import pandas as pd
import pyloudnorm as pln
import soundfile as sf
import shutil

from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw, sync_via_mrmsdtw_with_anchors
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic
from synctoolbox.feature.csv_tools import read_csv_to_df, df_to_pitch_features, df_to_pitch_onset_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

from alignment_dtw_utils import *
from paths import *
from musicxml_parser.csv_to_midi import df_to_midi


# some synctoolbox parameters
Fs = 22050
feature_rate = 50
step_weights = np.array([1.5, 1.5, 2.0])
threshold_rec = 10 ** 6

# If True sonifies the alignemnt and saves an audio file in the csv dir
sonification = True
# Path for sonification output
sonification_dir = os.path.join(bsed_dir, "04_Extra", "Sequence-Alignment_Sonification")

#Path of Audio recordings
audio_dir = os.path.join(bsed_dir, "02_Audio", "wav_44100_440Hz")

#Path of Annotations
score_dir = os.path.join(bsed_dir, "01_ScoreData", "CSV")

#Path for Start-End Annotations
anchor_dir = os.path.join(bsed_dir, "04_Extra", "anchor_points")



#Path for Alignment output
csv_out_dir = os.path.join(bsed_dir, "03_NoteAnnotations", "Sequence-Alignment")

transcription_dir = os.path.join(bsed_dir, "04_Extra", "transcriptions_MN-Transcriber")

use_transcription_features = True
override = True

 
def orch_sync(audio_path, ann_path, anchor_path, save_dir, sonification_dir=None,
               Fs=Fs, feature_rate=feature_rate, step_weights=step_weights, threshold_rec = threshold_rec, sonification=sonification):
 
    #Load audio
    audio, _ = librosa.load(audio_path, sr=Fs, mono=True)

    #Load annotation
    df_annotation = pd.read_csv(ann_path, sep=";", index_col="index")
    df_annotation.rename(columns={"start_sec": "start", "end_sec": "end", "duration_sec" : "duration"}, inplace=True)
    #Making the ann start point equal to audio start
    
    ann_start = np.min(df_annotation["start"])
    if ann_start!=0:
        df_annotation["end"] -= ann_start
        df_annotation["start"] -= ann_start

        
    #"stretch" the csv-notes to match the length of the audio recording
    df_annotation["duration"] = df_annotation["end"] - df_annotation["start"]
    annotation_dur = np.max(df_annotation["start"] + df_annotation["duration"])-np.min(df_annotation["start"])
    annotation_stretch_fac = (len(audio)/Fs) / annotation_dur

    df_annotation["start"] *= annotation_stretch_fac
    df_annotation["duration"] *= annotation_stretch_fac
    df_annotation["end"] *= annotation_stretch_fac

    df_annotation["start"] += 0.021 # skip first frame to loosen start matching criteria of DTW
    df_annotation["end"] += 0.021


    
    #Computing quantized chroma and DLNCO features
    if use_transcription_features:
        print(f"Using transcription features for {audio_path}")
        transcription_features = np.load(os.path.join(transcription_dir, os.path.basename(audio_path).replace(".wav", ".npz")))
        f_chroma_quantized_audio, f_DLNCO_audio = get_features_from_transcription(transcription_features=transcription_features,
                                                                                  audio_length_seconds=len(audio)/Fs,
                                                                                  feature_rate=feature_rate)
    else:
        tuning_offset = estimate_tuning(audio, Fs)
        print('Estimated tuning deviation for recording: %d cents' % (tuning_offset))
        f_chroma_quantized_audio, f_DLNCO_audio = get_features_from_audio(audio, tuning_offset, Fs, feature_rate)
    f_chroma_quantized_annotation, f_DLNCO_annotation = get_features_from_annotation(df_annotation, feature_rate)

    #Finding optimal shift of chroma vectors
    f_cens_1hz_audio = quantized_chroma_to_CENS(f_chroma_quantized_audio, 201, 50, feature_rate)[0]
    f_cens_1hz_annotation = quantized_chroma_to_CENS(f_chroma_quantized_annotation, 201, 50, feature_rate)[0]
    
    opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_audio, f_cens_1hz_annotation)
    assert opt_chroma_shift == 0, "Found piece with chroma shift"
    print('Pitch shift between the audio recording and score, determined by DTW:', opt_chroma_shift, 'bins')
    
    f_chroma_quantized_annotation = shift_chroma_vectors(f_chroma_quantized_annotation, opt_chroma_shift)
    f_DLNCO_annotation = shift_chroma_vectors(f_DLNCO_annotation, opt_chroma_shift)


    if anchor_path is not None and os.path.exists(anchor_path):
        # Create mapping and merge in one go
        print(f"Using anchor points for {audio_path}")
        df_anchor = pd.read_csv(anchor_path, sep="\t", names = ["time", "_", "measure"])
        mapping = df_annotation.drop_duplicates('start_meas').set_index('start_meas')['start']
        df_anchor['start'] = df_anchor['measure'].map(mapping)
        anchor_points = list(zip(df_anchor['time'], df_anchor['start']))
    else:
        anchor_points = None

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

    #np.save(os.path.join(save_dir,os.path.basename(audio_path).replace(".wav", "_warping_path.npy")), wp)

    df_annotation_warped = df_annotation.copy(deep=True)

    if anchor_points is not None:
        # In order to enforce anchor points strictly, we have to quantize the time for the "anchored" notes 
        mask = df_annotation["start_meas"].isin(df_anchor["measure"].unique())
        df_annotation.loc[mask, 'start'] = (df_annotation.loc[mask, 'start'] * feature_rate).astype(int) / feature_rate
        
        mask = df_annotation["end_meas"].isin(df_anchor["measure"].unique())
        df_annotation.loc[mask, 'end'] = (df_annotation.loc[mask, 'end'] * feature_rate).astype(int) / feature_rate


    df_annotation_warped[['start', 'end']] = scipy.interpolate.interp1d(wp[1] / feature_rate,
                                wp[0] / feature_rate, kind='linear', fill_value="extrapolate")(df_annotation[['start', 'end']])
    
    df_annotation_warped.rename(columns={"start": "start_sec", "end": "end_sec", "duration" : "duration_sec"}, inplace=True)
    df_annotation_warped["duration_sec"] = df_annotation_warped["end_sec"] - df_annotation_warped["start_sec"]
    df_annotation_warped["start_sec"] = np.round(df_annotation_warped['start_sec'], 3)
    df_annotation_warped["end_sec"] = np.round(df_annotation_warped['end_sec'], 3)
    df_annotation_warped["duration_sec"] = np.round(df_annotation_warped['duration_sec'], 3)
    
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    #Define the path to save the CSV file

    csv_filepath = os.path.join(save_dir, f'{base_filename}.csv')

    if sonification:
        if sonification_dir is None:
            sonification_dir = save_dir
        sonify(audio, df_annotation_warped, sr=Fs, out_path=os.path.join(sonification_dir, f'{base_filename}.wav'))
        
    #Save the DataFrame as a CSV file
    df_annotation_warped.to_csv(csv_filepath, sep=';', index=True, index_label="index")
    mid = df_to_midi(df_annotation_warped)
    mid.write(csv_filepath.replace(".csv", ".mid"))

    
def main():
    os.makedirs(csv_out_dir, exist_ok=True)
    if sonification:
        os.makedirs(sonification_dir, exist_ok=True)
    audio_list = [f for f in sorted(os.listdir(audio_dir)) if ("Beethoven" in f and ".wav" in f)]
    audio_path_list = [os.path.join(audio_dir,f) for f in audio_list]


    for audio_path in tqdm(audio_path_list):

        if "Synth" in audio_path:
            # we don't have to align the synthetic audio
            continue

        file_name = os.path.basename(audio_path)
        bsed_id = file_name.split("_")[0]
        composer = file_name.split("_")[2]
        mvmt_id = file_name.split("_")[3]

        ann_path = os.path.join(score_dir,f"{bsed_id}_{composer}_{mvmt_id}.csv")
        anchor_path = os.path.join(anchor_dir, file_name.replace(".wav", ".txt"))

        out_path = os.path.join(csv_out_dir, file_name.replace(".wav", ".csv"))

        if os.path.exists(out_path) and not override:
            print("Already synced", file_name)
        else:
            orch_sync(audio_path, ann_path, anchor_path, csv_out_dir,
                    Fs=Fs, feature_rate=feature_rate, step_weights=step_weights, threshold_rec = threshold_rec,
                    sonification_dir=sonification_dir)

if __name__ == "__main__":
    main()
