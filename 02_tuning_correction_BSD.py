import os
import pyrubberband as pyrb
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from libfmp.c3 import compute_freq_distribution, tuning_similarity
from paths import *

bsd_audio_dir = os.path.join(bsd_dir, "02_Audio", "wav_44100")
bsd_audio_dir_440 = os.path.join(bsd_dir, "02_Audio", "wav_44100_440Hz")
tuning_file = os.path.join(bsd_dir, "04_Extra", "tuning_estimates.csv")
transposition_file = os.path.join(bsd_dir, "04_Extra", "transpositions.csv")

override = False

# Override only the files for which there is a transposition estimate available
# This is useful for running the script multiple times after finding transposition errors in the evaluation later on
override_only_transp = False



def libfmp_estimate_tuning(y, sr):
    v, _ = compute_freq_distribution(y, Fs=sr)
    _, _, _, theta_max, _ = tuning_similarity(v)
    return int(theta_max)


def main():
    if not os.path.isdir(bsd_audio_dir):
        raise FileNotFoundError(f"Did not find BSD audio dir: {bsd_audio_dir}")
    os.makedirs(bsd_audio_dir_440, exist_ok=True)
    audio_files = [f for f in os.listdir(bsd_audio_dir) if f.endswith(".wav")]
    df = pd.read_csv(tuning_file, sep=";", index_col="filename")
    df_transposition = pd.read_csv(transposition_file, sep=";", index_col="filename")
    for f in tqdm(audio_files):
        if os.path.exists(os.path.join(bsd_audio_dir_440, f)) and not override:  
            if f in df_transposition.index and override_only_transp:
                pass
            else:
                print(f"Found existing pitch shifted version, skipping")
                continue
        if f in df.index:
            tuning = df.loc[f, "tuning_libfmp"]
        else:
            y, sr = sf.read(os.path.join(bsd_audio_dir, f))
            tuning = libfmp_estimate_tuning(y, sr)
            df.loc[f, "tuning_libfmp"] = tuning
        if f in df_transposition.index:
            transp = df_transposition.loc[f, "transposition"]
        else:
            transp = 0

        pitch_shift_semitones = -(tuning/100 + transp)
        y, sr = sf.read(os.path.join(bsd_audio_dir, f))
        y_shifted = pyrb.pitch_shift(y, sr=44100, n_steps = pitch_shift_semitones)
        sf.write(os.path.join(bsd_audio_dir_440, f), y_shifted, 44100)

    df.to_csv(tuning_file, sep=";")

if __name__ == "__main__":
    main()