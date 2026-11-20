import os
import music21
import pandas as pd
from libfmp import b
import numpy as np
from music21 import converter
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import pretty_midi
import warnings
try:
    from .instrument_catalog import (
        instrument_in_catalog,
        resolve_ambiguous_instrument_name,
    )
except ImportError:
    from instrument_catalog import (
        instrument_in_catalog,
        resolve_ambiguous_instrument_name,
    )

OUTPUT_COLUMNS = ["start_meas",
                  "end_meas",
                  "start_quarter",
                  "duration_quarter",
                  "time_sig",
                  "pitch",
                  "pitch_name",
                  "part",
                  "instrument",
                  "articulation",
                  "expression",
                  "dynamic",
                  "tempo_qpm",
                  "start_sec",
                  "end_sec",
                  "duration_sec"]

OUTPUT_COLUMNS_DTYPES = {
    "start_meas": "float64",
    "end_meas": "float64",
    "start_quarter": "float64",
    "duration_quarter": "float64",
    "time_sig": "string",
    "pitch": "Int64",
    "pitch_name": "string",
    "part": "string",
    "instrument": "string",
    "articulation": "string",
    "expression": "string",
    "dynamic": "string",
    "tempo_qpm": "float64",
    "start_sec": "float64",
    "end_sec": "float64",
    "duration_sec": "float64",
}

INSTRUMENT_SOURCE_CHOICES = ("score_name", "midi", "part_name")
"""
Instrument source options:
- score_name: Use the instrument name from the score (music21's instrumentName attribute), often implicitly set by the score editing programs
- midi: Use the MIDI program number and channel to determine the instrument name. If the channel is 9, it is considered unpitched percussion. Often set by the playback options of the Sheet Music Program
- part_name: Use the part name from the score and try to clean it, i.e., remove transposition and part numbers.
"""
DEFAULT_INSTRUMENT_SOURCE = "part_name"
TRILL_EXPRESSION_TYPES = [music21.expressions.Trill]
if hasattr(music21.expressions, "TrillExtension"):
    TRILL_EXPRESSION_TYPES.append(music21.expressions.TrillExtension)
TRILL_EXPRESSION_TYPES = tuple(TRILL_EXPRESSION_TYPES)


def get_instrument_from_part_name(part_name):
    """
    Clean part name by removing transpositon information, extra whitespace, and any part numbering
    """
    if part_name is None:
        return ""
    
    # Normalize whitespace
    name = " ".join(part_name.strip().split())
    
    # Remove transposition information (e.g., "in G", "in B♭", "in E♭")
    import re
    name = re.sub(r'\s+in\s+[A-G].*?(?=\s|$)', '', name, flags=re.IGNORECASE)
    
    # Remove part numbering (Arabic numerals or Roman numerals at the end)
    name = re.sub(r'\s+(?:[0-9]+|[IVivx]+)\s*$', '', name)
    
    # check if any remaining part of the name is a single letter (e.g., "Flute A" -> "Flute")
    name_parts = name.split()
    if len(name_parts) > 1 and len(name_parts[-1]) == 1:
        name = " ".join(name_parts[:-1])

    # Clean up any remaining extra whitespace
    name = " ".join(name.strip().split())

    name = resolve_ambiguous_instrument_name(name)
    
    return name

def validate_instrument_source(instrument_source):
    if instrument_source is None:
        # default
        return DEFAULT_INSTRUMENT_SOURCE

    if instrument_source not in INSTRUMENT_SOURCE_CHOICES:
        raise ValueError(
            f"Invalid instrument_source '{instrument_source}'. "
            f"Choose one of {INSTRUMENT_SOURCE_CHOICES}."
        )

    return instrument_source


def resolve_instrument(part, instrument_source):
    instrument_source = validate_instrument_source(instrument_source)

    if instrument_source == "part_name":
        return get_instrument_from_part_name(part.partName)

    elif instrument_source == "midi":
        midi_program = part.getInstrument().midiProgram
        midi_channel = part.getInstrument().midiChannel
        if midi_channel == 9:
            return "Unpitched Percussion"

        if midi_program is None:
            return ""

        try:
            return pretty_midi.program_to_instrument_name(int(midi_program))
        except (TypeError, ValueError):
            return ""
    elif instrument_source == "score_name":
        instrument = part.getInstrument()
        return getattr(instrument, "instrumentName", "")
    else:
        raise ValueError(f"Invalid instrument_source '{instrument_source}'.")


def warn_unknown_instruments(df):
    unique_instruments = sorted(
        {str(name).strip() for name in df["instrument"].dropna().tolist() if str(name).strip()}
    )
    for instrument_name in unique_instruments:
        if not instrument_in_catalog(instrument_name):
            warnings.warn(
                f"Instrument '{instrument_name}' is not in instrument_catalog.csv.",
                stacklevel=2,
            )


def resolve_dynamic(note):
    dynamic_context = None

    if hasattr(note.volume, "getDynamicContext"):
        try:
            dynamic_context = note.volume.getDynamicContext()
        except Exception:
            dynamic_context = None

    if dynamic_context is None:
        try:
            dynamic_context = note.getContextByClass(music21.dynamics.Dynamic)
        except Exception:
            dynamic_context = None

    if dynamic_context is None:
        return ""

    if hasattr(dynamic_context, "value") and dynamic_context.value:
        return dynamic_context.value

    if hasattr(dynamic_context, "name") and dynamic_context.name:
        return dynamic_context.name

    return ""


def is_unpitched_note(note):
    return isinstance(note, music21.note.Unpitched)


def realize_tremolo_if_present(note):
    """
    Returns a list of notes. If tremolo present, returns realized notes; otherwise returns [note].
    """
    # Check for tremolo in expressions
    tremolo = None
    for expr in note.expressions:
        if isinstance(expr, music21.expressions.Tremolo):
            tremolo = expr
            break

    if tremolo is None:
        return [(note, 0)]

    # Use music21's built-in realize function
    # Returns tuple: (realized_notes, original_note, additional_notes)
    realized_notes, _, _ = tremolo.realize(note)
    # Calculate cumulative offsets for each realized note
    result = []
    cumulative_offset = 0.0
    for realized_note in realized_notes:
        result.append((realized_note, cumulative_offset))
        cumulative_offset += realized_note.duration.quarterLength

    return result


def realize_trill_if_present(note):
    """Realize trill expressions on a note and return (note, offset) tuples in quarter lengths."""
    trill = None
    for expr in note.expressions:
        if isinstance(expr, TRILL_EXPRESSION_TYPES):
            trill = expr
            break

    if trill is None:
        return [(note, 0)]

    try:
        realization = trill.realize(note)
    except Exception as exc:
        print(f"Failed to realize trill for note {note!r}: {exc}")
        return [(note, 0)]

    if isinstance(realization, tuple):
        realized_notes = realization[0]
    elif hasattr(realization, "recurse"):
        realized_notes = list(realization.recurse().notes)
    elif isinstance(realization, (list, tuple)):
        realized_notes = list(realization)
    else:
        realized_notes = [realization]

    if not realized_notes:
        return [(note, 0)]

    result = []
    cumulative_offset = 0.0
    for realized_note in realized_notes:
        result.append((realized_note, cumulative_offset))
        cumulative_offset += realized_note.duration.quarterLength

    return result


def get_pitches(note):
    """
    Extract pitches from note/chord.
    """
    if note.isChord:
        return list(note.pitches)
    else:
        if is_unpitched_note(note):
            return [music21.pitch.Pitch(-1)]
        else:
            return [note.pitch]


def format_pitch_name_scientific(pitch):
    """Return scientific pitch notation using # and b accidentals."""
    if pitch is None:
        return ""

    step = pitch.step or ""
    accidental = ""
    if pitch.accidental is not None:
        accidental = (pitch.accidental.modifier or "").replace("-", "b")

    octave = "" if pitch.octave is None else str(pitch.octave)
    return f"{step}{accidental}{octave}"

def parse_part(part, df, instrument_source, realize_tremolo=True, realize_trill=True):

    instrument = resolve_instrument(part, instrument_source)


    single_rows = []
    # Main loop
    for note in part.toSoundingPitch().recurse().notes:
        measure = note.measureNumber
        timeSig = note._getTimeSignatureForBeat().numerator / note._getTimeSignatureForBeat().denominator
        beatDur = note.beatDuration.quarterLength / 4
        beat = note.beat - 1
        beatNr = beat * beatDur

        articulation = "_".join([art.name for art in note.articulations])
        
        expression = "_".join([exp.name for exp in note.expressions])

        # Realize tremolo or trill, assuming the same note never carries both simultaneously
        has_tremolo = any(isinstance(exp, music21.expressions.Tremolo) for exp in note.expressions)
        has_trill = any(isinstance(exp, TRILL_EXPRESSION_TYPES) for exp in note.expressions)

        if realize_tremolo and has_tremolo:
            note_list = realize_tremolo_if_present(note)
        elif realize_trill and has_trill:
            note_list = realize_trill_if_present(note)
        else:
            note_list = [(note, 0)]
        # note_list is a list of tuples (realized_note, time_offset_ql),
        # where time_offset_ql is the offset in quarter lengths from the original note's start time

        for realized_note, time_offset_ql in note_list:
            noteDur = realized_note.duration.quarterLength / 4
            adjusted_beatNr = beatNr + (time_offset_ql / 4)

            note_is_unpitched = is_unpitched_note(realized_note)
            for pitch in get_pitches(realized_note):
                pitch_value = -1 if note_is_unpitched else pitch.midi
                pitch_name = "Unpitched" if note_is_unpitched else format_pitch_name_scientific(pitch)
                row = {
                    "start_meas": np.round(measure + adjusted_beatNr / timeSig, 3),
                    "end_meas": np.round(measure + (adjusted_beatNr + noteDur) / timeSig, 3),
                    "duration_quarter": max(noteDur * 1.0, 1 / 16),
                    "pitch": pitch_value,
                    "pitch_name": pitch_name,
                    "part": part.partName,
                    "instrument": instrument,
                    "time_sig": f"{realized_note._getTimeSignatureForBeat().numerator}/"
                                f"{realized_note._getTimeSignatureForBeat().denominator}",
                    "articulation": articulation,
                    "expression": expression,
                    "dynamic": resolve_dynamic(realized_note),
                    "start_quarter": np.nan,
                    "tempo_qpm": np.nan,
                }

                single_rows.append(row)
                if row['duration_quarter'] == 0:
                    print(row)
                    break

    df_ = pd.DataFrame(single_rows)
    df_.sort_index(inplace=True)
    df = pd.concat([df, df_])
    return df


def calc_quarterNoteOffset(df):
    """
    Calculate quarter note offset, i.e., how many quarter notes we have to count from the beginning to arrive at the
     given note event
    :param df:
    :return:
    """

    quarterNoteOffset = []
    beat_fac = 1
    curr_meas = 0.
    meas_offset = 0

    beat_fac_list = []
    quarterNoteOffset_in_measure = []

    for i, row in df.iterrows():

        # if new measure starts
        if np.floor(row.start_meas) > curr_meas:
            curr_meas = np.floor(row.start_meas)
            meas_offset += beat_fac * 4

        beat_fac = int(row.time_sig.split("/")[0]) / int(row.time_sig.split("/")[1])
        quarterNoteOffset.append(meas_offset + (row.start_meas - curr_meas) * beat_fac * 4)
        quarterNoteOffset_in_measure.append((row.start_meas - curr_meas) * beat_fac * 4)

        beat_fac_list.append(beat_fac)

    df["start_quarter"] = quarterNoteOffset
    df["start_quarter"] -= min(df["start_quarter"])

    return df


def calc_qbpm(score, df):
    tempoMarks = [(score.beatAndMeasureFromOffset(st)[1].measureNumber, tmp.getQuarterBPM()) for st, en, tmp in
                  score.metronomeMarkBoundaries()]
    if len(tempoMarks) == 0:
        print("No tempo marks found")
        df["tempo_qpm"] = np.nan
        return df

    quarterNoteBPM = []
    for _, row in df.iterrows():
        qbpm = tempoMarks[0][1]
        for st, tmp in tempoMarks:
            if row.start_meas >= st:
                qbpm = tmp
        quarterNoteBPM.append(qbpm)
    df["tempo_qpm"] = quarterNoteBPM
    return df

def calculate_physical_time(df):

    if df["tempo_qpm"].isnull().all():
        print("No tempo marks found, cannot calculate physical time")
        df["start_sec"] = np.nan
        df["end_sec"] = np.nan
        df["duration_sec"] = np.nan
        return df
    changes = df[df["tempo_qpm"] != df["tempo_qpm"].shift()]

    # Create a list of tuples (start_meas, quarter_noteBPM)
    tempoMarks = list(zip((changes["start_meas"]).astype(int), changes["tempo_qpm"].astype(int)))
    tempoMarks.append((int(np.ceil(max(df["end_meas"]))), tempoMarks[-1][-1]))

    # tempo marks is a list with (measure_number, quarte_bpm) for each tempo change
    # Does this mean that a tempo change can/should only occur at measure boundaries?
    tempoMap_meas = []
    tempoMap_qbpm = []
    tempoMap_timeSig = []
    last_timeSig = None

    for i in range(len(tempoMarks) - 1):
        st, qbpm = tempoMarks[i]
        en = tempoMarks[i + 1][0]

        for meas in range(st, en):
            tempoMap_meas.append(meas)
            tempoMap_qbpm.append(qbpm)

            notes_meas = df[df["start_meas"].astype(int) == meas]

            if len(notes_meas) > 0:
                last_timeSig = list(notes_meas["time_sig"])[0]
                tempoMap_timeSig.append(last_timeSig)
            elif meas == 0:
                continue
            else:
                tempoMap_timeSig.append(last_timeSig)

    tempoMap_startTime = []
    last_start_time = 0
    for meas, qbpm, timeSig in zip(tempoMap_meas, tempoMap_qbpm, tempoMap_timeSig):
        n_quarters = int(timeSig.split("/")[0]) / int(timeSig.split("/")[1]) * 4

        sec_per_meas = n_quarters / qbpm * 60

        tempoMap_startTime.append(last_start_time)
        last_start_time += sec_per_meas
    tempoMap_meas.append(tempoMap_meas[-1] + 1)
    tempoMap_startTime.append(last_start_time)
    start_times = np.interp(x=df["start_meas"], xp=tempoMap_meas, fp=tempoMap_startTime)
    end_times = np.interp(x=df["end_meas"], xp=tempoMap_meas, fp=tempoMap_startTime)

    # make sure that we start at t=0 (important for pieces with an anacrusis)
    offset = start_times[0]
    start_times = start_times - offset
    end_times = end_times - offset

    df["start_sec"] = np.round(start_times, 3)
    df["end_sec"] = np.round(end_times, 3)
    df["duration_sec"] = np.round(df["end_sec"] - df["start_sec"], 3)

    return df

def resolve_ambiguous_instrument_names(df):
    unique_instruments = df["instrument"].unique()
    for instrument in unique_instruments:
        instrument_name = resolve_ambiguous_instrument_name(instrument)
        if instrument_name != instrument:
            print(f"Ambiguous instrument name resolved: {instrument} -> {instrument_name}")
        df.loc[df["instrument"] == instrument, "instrument"] = instrument_name
    return df


def apply_output_dtypes(df):
    for col, dtype in OUTPUT_COLUMNS_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df

def xml_to_df(xml_path, expand_repeats=False, supress_anacrusis=False,
              calc_physical_time=True, realize_tremolo=True, realize_trill=True,
              instrument_source=None, suppress_instrument_warning=False):
    """
    Parse a music xml file with m21 and convert it to an unfolded score piano roll and return is as a pandas dataframe
    :param xml_path: Full path to the xml file
    :param expand_repeats: If true, use m21 expandRepeat() to "unfold" the score and get rid of repeats.
    Warning: Since m21 expandRepeats() can be sometimes problematic, the use of this option is discouraged and ideally
     the provided xml should be already unfolded
    :param supress_anacrusis: If true, don't subtract 1 from measure numbers
    :param realize_trill: If True, expand trill expressions into explicit notes.
    :param instrument_source: Required instrument resolution mode: 'instrument_name', 'midi' or 'part_name'
    :param suppress_instrument_warning: If True, suppress warnings for instruments
     that are not in instrument_catalog.csv.
    :return: Tuple containing the parsed score and a dataframe in the new CSV schema
    """
    instrument_source = validate_instrument_source(instrument_source)

    # parse with m21
    s = converter.parse(xml_path)

    # We can use m21 to expand repeats automatically, however this does not work reliably
    # therefore we do not recommend this option in general
    if expand_repeats:
        s = s.expandRepeats()

    # Ties are considered a single note event
    s = s.stripTies()

    df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = apply_output_dtypes(df)

    for p in tqdm(s.parts, desc="Parsing XML parts"):
        # parse each part individually
        df = parse_part(
            p,
            df,
            instrument_source=instrument_source,
            realize_tremolo=realize_tremolo,
            realize_trill=realize_trill,
        )

    df.sort_values(by=["start_meas", "part"], inplace=True)

    # Process Anacrusis
    if ((np.min(df["start_meas"]) - 1.0) > 0) and not supress_anacrusis:
        print("subtract 1 measure (Anacrusis)")
        df[["start_meas", "end_meas"]] = df[["start_meas", "end_meas"]].apply(lambda x: x - 1.0)

    df = calc_quarterNoteOffset(df)
    df = calc_qbpm(s, df)

    if calc_physical_time:
        df = calculate_physical_time(df)

    df = resolve_ambiguous_instrument_names(df)
    if not suppress_instrument_warning:
        warn_unknown_instruments(df)

    # avoid having the last note end on a non existing measure
    max_val = df['end_meas'].max()
    if float(max_val).is_integer():
        df.loc[df['end_meas'] == max_val, 'end_meas'] -= 0.001

    # filter and sort
    df = df[OUTPUT_COLUMNS]

    # round everything
    for col in ["start_meas", "end_meas", "start_quarter", "duration_quarter", "start_sec", "end_sec", "duration_sec", "tempo_qpm"]:
        df[col] = df[col].round(3)

    df = apply_output_dtypes(df)

    return df.copy()


def xml_to_csv(xml_path, output_path, **kwargs):
    """
    Parse xml file and save in to a csv file in output_path

    :param xml_path: Full path to the xml file
    :param output_path: Path to the output directory
    :param expand_repeats: See xml_to_df
    :param supress_anacrusis: If True, don't subtract 1 from measure numbers in any case.
     Set to True for writing out excerpts, where the first measure number of the excerpt is > 1
    :param instrument_source: See xml_to_df
    :param suppress_instrument_warning: See xml_to_df
    :return: Dataframe containing note events in the new CSV schema
    """
    if not os.path.isdir(output_path):
        raise ValueError(f"{output_path} is not a directory")
    df = xml_to_df(xml_path, **kwargs)
    filename = os.path.splitext(os.path.basename(xml_path))[0] + ".csv"
    df.to_csv(os.path.join(output_path, filename), index=False, sep=";", float_format="%.3f")
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Convert a MusicXML file to a CSV file."
    )
    parser.add_argument(
        "xml_path", type=str, help="Path to the input MusicXML file."
    )
    parser.add_argument(
        "output_path", type=str, help="Directory to save the CSV file."
    )
    parser.add_argument(
        "--instrument-source",
        choices=INSTRUMENT_SOURCE_CHOICES,
        default=DEFAULT_INSTRUMENT_SOURCE,
        help=f"Choose source for instrument naming (default: {DEFAULT_INSTRUMENT_SOURCE}).",
    )
    parser.add_argument(
        "--expand-repeats", action="store_true", help="Expand repeats in the score."
    )
    parser.add_argument(
        "--supress-anacrusis", action="store_true", help="Don't subtract 1 from measure numbers."
    )
    parser.add_argument(
        "--no-physical-time", action="store_true", help="Don't calculate physical time (seconds)."
    )
    parser.add_argument(
        "--no-realize-tremolo", action="store_true", help="Don't realize tremolo expressions."
    )
    parser.add_argument(
        "--no-realize-trill", action="store_true", help="Don't realize trill expressions."
    )
    parser.add_argument(
        "--suppress-instrument-warning",
        action="store_true",
        help="Suppress warnings for unknown instruments.",
    )

    args = parser.parse_args()

    xml_to_csv(
        args.xml_path,
        args.output_path,
        instrument_source=args.instrument_source,
        expand_repeats=args.expand_repeats,
        supress_anacrusis=args.supress_anacrusis,
        calc_physical_time=not args.no_physical_time,
        realize_tremolo=not args.no_realize_tremolo,
        realize_trill=not args.no_realize_trill,
        suppress_instrument_warning=args.suppress_instrument_warning,
    )
    print(f"CSV file saved to {args.output_path}")

if __name__ == "__main__":
    main()
