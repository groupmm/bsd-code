import os
from musicxml_parser.xml_to_csv import xml_to_df
from musicxml_parser.csv_to_midi import df_to_midi
from musicxml_parser.utils import ensure_velocity_column
import pretty_midi
from paths import *
from tqdm import tqdm
from music21 import environment
import pandas as pd


def handle_m21_cache():
     # Create a cache for music21 to avoid permission problems
    cache_dir = os.path.expanduser('~/.music21_cache')
    os.makedirs(cache_dir, exist_ok=True)
    env = environment.Environment()
    env['directoryScratch'] = cache_dir

def main():
    xml_dir = os.path.join(bsed_dir, "01_ScoreData", "MusicXML_sounding_pitch")
    csv_dir = os.path.join(bsed_dir, "01_ScoreData", "CSV")
    midi_dir = os.path.join(bsed_dir, "01_ScoreData", "MIDI")
    input_format = ".musicxml"
    override = True
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(midi_dir, exist_ok=True)

    handle_m21_cache()

    xml_paths = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith(input_format)]
    xml_paths.sort()

    for xml_path in tqdm(xml_paths, desc=f"Parsing all input files"):
        if (os.path.exists(os.path.join(csv_dir, os.path.basename(xml_path).replace(input_format, ".csv")))
                and not override):
            print(f"Found parsed {os.path.join(csv_dir, os.path.basename(xml_path).replace(input_format, '.csv'))}"
                  f", skipping")
            continue

        if "BSED-03" in xml_path:
            supress_anacrusis = False
        else:
            supress_anacrusis = True

        df = xml_to_df(xml_path, calc_physical_time=True, supress_anacrusis=supress_anacrusis, realize_tremolo=True, realize_trill=True)

        if "BSED-12" in xml_path:
            # manually fix some names for trombones
            for specifier in ["alto", "bass", "tenor"]:
                df["part"] = df["part"].replace(f"Trombone {specifier}", f"{specifier.capitalize()} Trombone")
                df["instrument"] = df["instrument"].replace(f"Trombone {specifier}", f"{specifier.capitalize()} Trombone")
        df = ensure_velocity_column(df)
        df.drop_duplicates(inplace=True)
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(csv_dir, os.path.basename(xml_path).replace(input_format, ".csv")),
                  sep=";", header=True, index=True, index_label="index",
                  float_format='%.3f')
        mid = df_to_midi(df)
        mid.write(os.path.join(midi_dir, os.path.basename(xml_path).replace(input_format, ".mid")))


if __name__ == "__main__":
    main()
