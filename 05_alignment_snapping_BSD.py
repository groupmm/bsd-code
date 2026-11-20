import numpy as np
from note_level_alignment import match_onsets_from_df
import torch
import pandas as pd
import os
from musicxml_parser.csv_to_midi import df_to_midi
import shutil
from tqdm import tqdm
import librosa
from paths import *
import time
from utils_bsd import get_peaks




features_dir = os.path.join(bsd_dir, "04_Extra", "transcriptions_MN-Transcriber")
seq_aligned_path = os.path.join(bsd_dir, "03_NoteAnnotations", "Sequence-Alignment")
out_path_snapped = os.path.join(bsd_dir, "03_NoteAnnotations", f"Note-Level-Alignment")
bsd_audio_dir = os.path.join(bsd_dir, "02_Audio", "wav_44100_440Hz")

# parameters from the transcription model
sr = 16000
hop_length = 512
min_midi = 21

neigbourhood_seconds = 1
threshold = 0.1 # notes with a matching onset probability below this threshold will be kept at their initial position

override = True

def main():
    pd.set_option("mode.copy_on_write", True)
    os.makedirs(out_path_snapped, exist_ok=True)
    unaligned_csv = [f for f in os.listdir(seq_aligned_path) if f.endswith(".csv")]
    for csv in tqdm(unaligned_csv):

        if os.path.exists(os.path.join(out_path_snapped, csv)) and not override:
            print(f"Found {os.path.join(out_path_snapped, csv)}, skipping")
            continue

        df = pd.read_csv(os.path.join(seq_aligned_path, csv), sep=";",index_col="index")
        df["duration_sec"] = df["end_sec"] - df["start_sec"]
        features = np.load(os.path.join(features_dir, csv.replace(".csv", ".npz")))
        onset_pred = features["onset_pred"]
        onset_pred = torch.from_numpy(onset_pred)
        onset_pred_mask = get_peaks(onset_pred, 1)
        onset_pred *= onset_pred_mask
        n_neighbours_frames = int(neigbourhood_seconds // (hop_length/sr))
        df = match_onsets_from_df(onset_pred, df, n_neighbours_frames,
                                        gaussian_weighting=False,
                                        pre_filter_multiplicity=True,
                                        post_filter_multiplicity=False,
                                        method="bipartite",
                                        min_midi=min_midi,
                                        sample_rate=sr,
                                        hop_length=hop_length, 
                                        reorder_alignments_by_measure_position = True,                                        
                                        return_score=True
                                            )
        
        # the matching score tells us how strong the activation in the transcription features is at the point where a note is matched to. 
        # If it is below the threshold, we assume that the transcriber didn't properly pick up the note and our best guess is to simply keep the initial alignemtn position
        df.loc[df["matching_score"] < threshold, ["start_sec", "end_sec"]] = df.loc[df["matching_score"] < threshold, ["start_initial", "end_initial"]].values
        df["matching_score"] = df["matching_score"].round(3)
        df.drop(columns=["start_initial", "end_initial"], inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(os.path.join(out_path_snapped, csv), index=True, index_label="index", sep=";", float_format='%.3f')
        mid = df_to_midi(df)
        mid.write(os.path.join(out_path_snapped, csv.replace(".csv", ".mid")))

if __name__ == "__main__":
    main()

