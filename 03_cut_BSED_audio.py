import os
import soundfile as sf
import pandas as pd
from paths import *

def cut_from_dir(bsd_audio_dir, bsed_audio_dir, df_bsed):
    for idx, row in df_bsed.iterrows():
        if "Blomstedt" in row["filename"] and not os.path.exists(os.path.join(bsd_audio_dir, row["filename"])):
            print("Can't cut audio from Blomstedt version as it's not in the full BSD audio directory, skipping.")
            continue
        audio_file = os.path.join(bsd_audio_dir, row["filename"])
        y, sr = sf.read(audio_file)
        assert sr == 44100, f"Found wrong sr {sr} for file {audio_file}"
        start_time, end_time = row["start"], row["end"]
        start_sample, end_sample = int(start_time*sr), int(end_time*sr)
        assert start_sample <= end_sample
        assert start_sample >= 0
        assert end_sample <= y.shape[0]
        y_cut = y[start_sample:end_sample, :]
        assert abs(y_cut.shape[0]/sr - (end_time-start_time)) < 0.001, f"{y_cut.shape[0]//sr}, {end_time-start_time}"
        new_audio_filename = row["bsedID"] + "_" + str(row["splitID"])+ "_" + row["filename"]
        sf.write(os.path.join(bsed_audio_dir, new_audio_filename), y_cut, sr)

def main():
    bsd_audio_dir = os.path.join(bsd_dir, "02_Audio", "wav_44100")
    assert os.path.exists(bsd_audio_dir), "Did not find BSD audio dir"
    bsed_audio_dir = os.path.join(bsed_dir, "02_Audio", "wav_44100")
    os.makedirs(bsed_audio_dir, exist_ok=True)

    df_bsed = pd.read_csv(os.path.join(bsed_dir, "04_Extra", "BSED_data.csv"), sep=";")

    cut_from_dir(bsd_audio_dir, bsed_audio_dir, df_bsed)

    # Do the same thing for the "tuning-corrected" audio
    bsd_audio_dir = os.path.join(bsd_dir, "02_Audio", "wav_44100_440Hz")
    assert os.path.exists(bsd_audio_dir), "Did not find BSD audio dir"
    bsed_audio_dir = os.path.join(bsed_dir, "02_Audio", "wav_44100_440Hz")
    os.makedirs(bsed_audio_dir, exist_ok=True)
    cut_from_dir(bsd_audio_dir, bsed_audio_dir, df_bsed)

if __name__ == "__main__":
    main()