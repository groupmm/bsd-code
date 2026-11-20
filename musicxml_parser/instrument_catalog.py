import os
import pandas as pd
import pretty_midi


CATALOG_PATH = os.path.join(os.path.dirname(__file__), "instrument_catalog.csv")

def validate_catalog():
    """
    Validate the instrument catalog for consistency and correctness.
    """
    df = pd.read_csv(CATALOG_PATH, sep=";", index_col="instrument")

    # Check for duplicate instrument names
    duplicates = df.index[df.index.duplicated()].unique()
    if not duplicates.empty:
        raise ValueError(f"Duplicate instrument names found in catalog: {', '.join(duplicates)}")
    
    # Check for duplicate abbreviation
    if any(df["abbreviation"].duplicated()):
        duplicates_abbr = df["abbreviation"][df["abbreviation"].duplicated()].unique()
        raise ValueError(f"Duplicate instrument abbreviation found in catalog: {', '.join(duplicates_abbr)}")

    # Check for missing values
    for col in df.columns:
        if df[col].isnull().any():
            raise ValueError(f"Missing values found in column '{col}' of the catalog.")

    # Check for valid MIDI program numbers (0-127)
    invalid_MIDI_programs = df[~df["MIDI_program"].between(0, 127)]
    if not invalid_MIDI_programs.empty:
        raise ValueError(f"Invalid MIDI program numbers found: {invalid_MIDI_programs['MIDI_program'].tolist()}")

    print("Instrument catalog validation passed.")

def get_instrument_catalog():
    return pd.read_csv(CATALOG_PATH, sep=";", index_col="instrument")

def get_instrument_family(instrument_name):
    df = pd.read_csv(CATALOG_PATH, sep=";", index_col="instrument")
    if instrument_name not in df.index:
        # Attempt to resolve ambiguous instrument names
        family = None
        print(f"Instrument '{instrument_name}' not found in catalog, can't resolve family.")
    else:
        family = df.loc[instrument_name, "instrument_family"]
    return family

def get_instrument_program(instrument_name):
    df = pd.read_csv(CATALOG_PATH, sep=";", index_col="instrument")
    instrument_name = resolve_ambiguous_instrument_name(instrument_name)
    if instrument_name not in df.index:
        # Attempt to resolve ambiguous instrument names
        try:
            program = pretty_midi.instrument_name_to_program(instrument_name)
        except ValueError:
            program = None
        print(f"Instrument '{instrument_name}' not found in catalog and not a general MIDI name, can't resolve program.")
    else:
        program = df.loc[instrument_name, "MIDI_program"]
    return program

def get_instrument_drum_flag(instrument_name):
    df = pd.read_csv(CATALOG_PATH, sep=";", index_col="instrument")
    instrument_name = resolve_ambiguous_instrument_name(instrument_name)
    if instrument_name not in df.index:
        is_drum = None
        print(f"Instrument '{instrument_name}' not found in catalog, can't resolve drum flag.")
    else:
        is_drum = df.loc[instrument_name, "is_unpitched_drum"]
    return is_drum

def get_instrument_abbreviation(instrument_name):
    df = pd.read_csv(CATALOG_PATH, sep=";", index_col="instrument")
    instrument_name = resolve_ambiguous_instrument_name(instrument_name)
    if instrument_name not in df.index:
        # Attempt to resolve ambiguous instrument names
        abbreviation = None
        print(f"Instrument '{instrument_name}' not found in catalog, can't resolve abbreviation.")
    else:
        abbreviation = df.loc[instrument_name, "abbreviation"]
    return abbreviation


def resolve_ambiguous_instrument_name(instrument_name):
    """
    Resolve ambiguous instrument names to standardized names. Also converts MIDI program names to our standardized names if applicable, e.g. 
    Acoustic Grand Piano -> Piano
    :param instrument_name: The instrument name to resolve.
    :return: The resolved instrument name.
    """
    if instrument_in_catalog(instrument_name):
        return instrument_name
    if instrument_name == "Horn":
        msg = "Found instrument 'Horn' in file. Assuming French Horn, if not, please specify manually."
        instrument_name = "French Horn"
    elif instrument_name == "Cello":
        msg = "Found instrument 'Cello' in file, adjusting to standardized name 'Violoncello'."
        instrument_name = "Violoncello"
    elif instrument_name == "Double Bass":
        msg = "Found instrument 'Double Bass' in file, adjusting to standardized name 'Contrabass'."
        instrument_name = "Contrabass"
    elif "Piano" in instrument_name and instrument_name != "Piano":
        msg = f"Found instrument '{instrument_name}' in file. Adjusting to standardized name 'Piano'."
        instrument_name = "Piano"
    elif "Harpsichord" in instrument_name and instrument_name != "Harpsichord":
        msg = f"Found instrument '{instrument_name}' in file. Adjusting to standardized name 'Harpsichord'."
        instrument_name = "Harpsichord"
    elif "Timpani" in instrument_name and instrument_name != "Timpani":
        msg = f"Found instrument '{instrument_name}' in file, simplifying to 'Timpani'."
        instrument_name = "Timpani"
    else:
        msg = None
    print(msg) if msg else None
    return instrument_name

def instrument_in_catalog(name):
    df = pd.read_csv(CATALOG_PATH, sep=";", index_col="instrument")
    
    is_in_catalog = name in df.index

    return is_in_catalog


if __name__ == "__main__":
    validate_catalog()
