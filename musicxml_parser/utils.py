import music21
import numpy as np

def dynamics_to_velocity(d: str) -> int:
    try:
        m21_dynamic = music21.dynamics.Dynamic(str(d).strip().lower())
        vel = m21_dynamic.volumeScalar * 127 if m21_dynamic.volumeScalar is not None else 64
        return int(max(1, min(127, vel)))
    except Exception:
        return 64
    
def ensure_velocity_column(df):
    """
    Ensures that the DataFrame has a "velocity" column.
    If "velocity" exists, it is clipped to the range [1, 127].
    If "volume" exists, it is converted to "velocity".
    If "dynamic" exists, it is converted to "velocity" using music21.
    Otherwise, "velocity" is set to 64.
    """
    if "velocity" in df.columns:
        df["velocity"] = df["velocity"].clip(1, 127).astype(int)
        return df

    if "volume" in df.columns:
        df["velocity"] = (df["volume"] * 127).clip(1, 127).astype(int)
        return df


    if "dynamic" in df.columns:
        df["velocity"] = df["dynamic"].map(dynamics_to_velocity)
        return df

    df["velocity"] = 64
    return df

