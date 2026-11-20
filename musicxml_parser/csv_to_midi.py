import argparse
import os

from music21 import dynamics
import pandas as pd
import pretty_midi

try:
    from .instrument_catalog import (
        get_instrument_program,
        get_instrument_drum_flag,
    )
    from .xml_to_csv import calculate_physical_time
except ImportError:
    from instrument_catalog import (
        get_instrument_program,
        get_instrument_drum_flag,
    )
    from xml_to_csv import calculate_physical_time


DEFAULT_VELOCITY = 80
DEFAULT_MIDI_PROGRAM = 0
DEFAULT_DRUM_MIDI_PITCH = 39 # DEFAULT to Hand Clap (MIDI note 39) for unpitched drums if no specific mapping is found
PIZZICATO_STRINGS_MIDI_PROGRAM = 45
STRING_INSTRUMENTS = {"Violin", "Viola", "Violoncello", "Double Bass"}


def _resolve_midi_program(instrument_name, is_pizzicato, is_drum):
    if is_pizzicato:
        return PIZZICATO_STRINGS_MIDI_PROGRAM

    program = get_instrument_program(instrument_name)
    if program is not None:
        return int(program)

    if instrument_name:
        try:
            return int(pretty_midi.instrument_name_to_program(instrument_name))
        except ValueError:
            pass

    if is_drum:
        print(f"Resorting to default drum MIDI pitch for instrument '{instrument_name}'")
        return DEFAULT_DRUM_MIDI_PITCH
    print(f"Resorting to default MIDI program {pretty_midi.program_to_instrument_name(DEFAULT_MIDI_PROGRAM)} for instrument '{instrument_name}'")
    return DEFAULT_MIDI_PROGRAM

def _is_drum_note(row):
    return int(row["pitch"]) == -1

def _is_pizzicato_note(row, instrument_name):
    if int(row["pitch"]) == -1 or instrument_name not in STRING_INSTRUMENTS:
        return False
    articulation_value = row.get("articulation", "")
    is_pizz = "pizzicato" in str(articulation_value).strip().lower()
    if is_pizz and instrument_name not in STRING_INSTRUMENTS:
        print(f"Warning: Found pizzicato articulation for non-string instrument '{instrument_name}'. Ignoring pizzicato.")
        return False
    return is_pizz


def ensure_velocity_column(df):
    if "velocity" in df.columns:
        df["velocity"] = df["velocity"].clip(1, 127).astype(int)
        return df

    if "volume" in df.columns:
        df["velocity"] = (df["volume"] * 127).clip(1, 127).astype(int)
        return df

    dynamic_col = next((c for c in ["dynamic", "dynamics"] if c in df.columns), None)
    if dynamic_col is not None:
        def dynamic_to_velocity(d):
            try:
                m21_dynamic = dynamics.Dynamic(str(d).strip().lower())
                vel = m21_dynamic.volumeScalar * 127 if m21_dynamic.volumeScalar is not None else DEFAULT_VELOCITY
                return int(max(1, min(127, vel)))
            except Exception:
                return DEFAULT_VELOCITY
        df["velocity"] = df[dynamic_col].map(dynamic_to_velocity)
        return df

    df["velocity"] = DEFAULT_VELOCITY
    return df


def _resolve_drum_pitch(instrument_name):
    program = get_instrument_program(instrument_name)
    if program is not None:
        return int(program)
    try:
        return pretty_midi.drum_name_to_note_number(instrument_name)
    except ValueError:
        return DEFAULT_DRUM_MIDI_PITCH


def _add_note_to_instrument(row, instrument, is_drum):
    start = float(row["start_sec"])
    end = float(row["end_sec"])
    if end <= start:
        print(f"Warning: Note with start time {start} and end time {end} has non-positive duration. Skipping this note.")
        return

    pitch = int(row["pitch"])
    if is_drum:
        pitch = _resolve_drum_pitch(str(row["instrument"]).strip())

    note = pretty_midi.Note(
        velocity=int(row["velocity"]),
        pitch=pitch,
        start=start,
        end=end,
    )
    instrument.notes.append(note)


def ensure_time_columns(df):
    # Priority 1: start_sec with end_sec or duration_sec
    if "start_sec" in df.columns:
        if "end_sec" in df.columns:
            return df
        if "duration_sec" in df.columns:
            df = df.copy()
            df["end_sec"] = df["start_sec"] + df["duration_sec"]
            return df

    # Priority 2: start with end or duration
    # in some old files we didn't use the _sec suffix, so we need to handle that case as well
    if "start" in df.columns:
        df = df.copy()
        df["start_sec"] = df["start"]
        if "end" in df.columns:
            df["end_sec"] = df["end"]
            return df
        if "duration" in df.columns:
            df["end_sec"] = df["start"] + df["duration"]
            return df

    # Priority 3: calculate physical time only when tempo_qpm exists and is not all NaN
    if "tempo_qpm" in df.columns:
        return calculate_physical_time(df)

    raise ValueError(
        "Could not resolve timing. Expected start_sec with end_sec/duration_sec, "
        "or start with end/duration, or tempo_qpm with valid values for time calculation."
    )


def _build_track_name(instrument_name, is_pizzicato, is_drum):
    if is_pizzicato:
        base = instrument_name or "Strings"
        return f"{base} Pizzicato"
    if is_drum:
        return instrument_name or "Drums"
    return instrument_name or pretty_midi.program_to_instrument_name(DEFAULT_MIDI_PROGRAM)


def _create_instrument(instrument_name, is_pizzicato, is_drum_note):
    program = _resolve_midi_program(instrument_name, is_pizzicato, is_drum_note)

    if is_pizzicato:
        midi_is_drum = False
    elif is_drum_note:
        midi_is_drum = True
    else:
        drum_flag = get_instrument_drum_flag(instrument_name)
        midi_is_drum = bool(drum_flag) if drum_flag is not None else False

    return pretty_midi.Instrument(
        program=program,
        is_drum=midi_is_drum,
        name=_build_track_name(instrument_name, is_pizzicato, is_drum_note),
    )


def df_to_midi(df):
    df = ensure_time_columns(df.copy())
    df = ensure_velocity_column(df)

    required_no_na_columns = ["start_sec", "end_sec", "instrument", "pitch"]
    for col in required_no_na_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values, which is not allowed.")

    if "instrument" not in df.columns:
        df["instrument"] = ""

    midi = pretty_midi.PrettyMIDI()
    midi_instruments = {}

    for _, row in df.iterrows():
        instrument_name = str(row["instrument"]).strip()
        is_drum_note = _is_drum_note(row)
        is_pizzicato_note = _is_pizzicato_note(row, instrument_name)

        track_key = (instrument_name, is_pizzicato_note, is_drum_note)

        if track_key not in midi_instruments:
            midi_instruments[track_key] = _create_instrument(
                instrument_name,
                is_pizzicato_note,
                is_drum_note,
            )

        _add_note_to_instrument(row, midi_instruments[track_key], is_drum_note)

    midi.instruments.extend(midi_instruments.values())
    return midi

def csv_to_midi(csv_path, output_path, sep=";"):
    """
    Parse CSV file and save it as a MIDI file in output_path.

    :param csv_path: Full path to the CSV file
    :param output_path: Path to output directory
    :return: pretty_midi.PrettyMIDI object
    """
    if not os.path.isdir(output_path):
        raise ValueError(f"{output_path} is not a directory")

    df = pd.read_csv(csv_path, sep=sep, engine="python")
    midi = df_to_midi(df)

    filename = os.path.splitext(os.path.basename(csv_path))[0] + ".mid"
    midi.write(os.path.join(output_path, filename))
    return midi



def main():
    parser = argparse.ArgumentParser(
        description="Convert a CSV file to MIDI format."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Full path to the input CSV file"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output directory where the MIDI file will be saved"
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=";",
        help="Field separator in the CSV file (default: ';')"
    )

    args = parser.parse_args()

    try:
        midi = csv_to_midi(args.csv_path, args.output_path, sep=args.sep)
        print(f"Successfully converted to MIDI with {len(midi.instruments)} instrument(s)")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit(1)

if __name__ == "__main__":
    main()
