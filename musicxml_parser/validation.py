import music21
import numpy as np
import pandas as pd

try:
    from .xml_to_csv import OUTPUT_COLUMNS
    from .instrument_catalog import instrument_in_catalog, resolve_ambiguous_instrument_name
except ImportError:
    from xml_to_csv import OUTPUT_COLUMNS
    from instrument_catalog import instrument_in_catalog, resolve_ambiguous_instrument_name

def validate_existing_df(df, raise_error_or_warning=False):
    required_columns = OUTPUT_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    critical_error = False # used for hard enforced rules
    warning_error = False # used for soft enforced rules/recommendations
    
    if missing_columns:
        print(f"Warning: The following required columns are missing from the DataFrame: {missing_columns}")
        critical_error = True
    for col in required_columns:
        if col in missing_columns: continue
        if col == "part":
            # check if roman numerals are present in the part column
            roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
            if df[col].apply(lambda x: any(s in roman_numerals for s in x.split())).any():
                print(f"Warning: The 'part' column likely contains roman numerals, while arabic numbers are preferred")
                df["roman_numeral_in_part"] = df[col].apply(lambda x: any(s in roman_numerals for s in str(x).split()))
                print(f"Parts with roman numerals: {df.loc[df['roman_numeral_in_part'], 'part'].unique()}")
                warning_error = True
        elif col == "instrument":
            if df[col].apply(lambda x: not instrument_in_catalog(x)).any():
                print(f"Warning: The 'instrument' column contains instruments not in the catalog")
                df["unknown_instrument"] = df[col].apply(lambda x: not instrument_in_catalog(x))
                unknown_instruments = df.loc[df['unknown_instrument'], 'instrument'].unique()
                print(f"Unknown instruments: {unknown_instruments}")
                for instr in unknown_instruments:
                    resolved_name = resolve_ambiguous_instrument_name(instr)
                    if resolved_name != instr:
                        print(f"Instrument '{instr}' should be resolved to '{resolved_name}'")
                warning_error = True
        elif col == "pitch_name":
            pitch_names = df[col].dropna().unique()
            if any(("-" in str(pn) or "+" in str(pn)) for pn in pitch_names):
                print(f"Warning: The 'pitch_name' column contains '-' and/or '+' characters, please use '#' and 'b' instead")
                critical_error = True
        
    if raise_error_or_warning:
        if critical_error:
            raise ValueError(f"Critical error(s) found in DataFrame validation. See warnings above.")
        if warning_error:
            raise Warning(f"Warning(s) found in DataFrame validation. See warnings above.")
    else:
        if critical_error:
            print(f"Critical error(s) found in DataFrame validation. See warnings above.")
        if warning_error:
            print(f"Warning(s) found in DataFrame validation. See warnings above.")
    if not critical_error and not warning_error:
        print(f"DataFrame validation passed with no errors or warnings.")

def validate_csv_file(csv_path, raise_error_or_warning=False):
    df = pd.read_csv(csv_path, sep=";")
    validate_existing_df(df, raise_error_or_warning=raise_error_or_warning)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate a CSV file generated from MusicXML.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file to validate.")
    parser.add_argument("--raise-error-or-warning", action="store_true", help="Raise an error or warning if validation fails.")
    args = parser.parse_args()
    
    validate_csv_file(args.csv_path, raise_error_or_warning=args.raise_error_or_warning)