import numpy as np
import pandas as pd
import scipy
from scipy.sparse.csgraph import min_weight_full_bipartite_matching as bipartite_match
from scipy.sparse import csr_array


EPSILON = 1e-4

def prepare_note_dataframe(df_notes, frame_length):
    df_notes = df_notes.copy()
    df_notes["onset_frame"] = np.floor(df_notes["start_sec"] / frame_length).astype(int)
    if "duration" not in df_notes.columns:
        assert "end_sec" in df_notes.columns, "Dataframe must contain either 'duration_sec' or 'end_sec' column"
        df_notes["duration_sec"] = df_notes["end_sec"] - df_notes["start_sec"]
    df_notes["duration_frame"] = np.round(df_notes["duration_sec"]/frame_length).astype(int)
    return df_notes

def finalize_aligned_dataframe(df_notes, frame_length):
    df_notes = df_notes.rename(columns = {"start_sec" : "start_initial", "end_sec": "end_initial"})
    df_notes = (
        df_notes[df_notes["aligned_onset_frame"].notna()]
        .assign(
            start_sec=lambda df: np.round(df["aligned_onset_frame"] * frame_length + frame_length / 2, 3),
            duration_sec=lambda df: np.round(df["end_initial"] - df["start_initial"], 3),
            end_sec=lambda df: np.round(df["start_sec"] + df["duration_sec"], 3)
        )
    )
    df_notes.drop(columns=["aligned_onset_frame", "duration_frame", "onset_frame"], inplace=True)
    return df_notes


def greedy_match(cost_matrix, neg_threshold):
    """
    Greedy match for a single pitch based on the cost_matrix
    cost_matrix has shape (n_frames, n_notes)
    """
    # Filter out frames that already have nearby aligned onsets
    n_frames, n_onsets = cost_matrix.shape

    mapping_rows = []
    mapping_cols = []
    aligned_frames_used = np.zeros(n_frames, dtype=bool)

    for n in range(n_onsets):
        cost_frames = cost_matrix[:, n]
        cost_frames[aligned_frames_used] = 0
        
        # Select frame with highest onset prediction

        aligned_frame = np.argmax(cost_frames)
        # Apply threshold filter
        if cost_frames[aligned_frame] < neg_threshold:
            print(f"Hit negative threshold")
            continue
        
        # Mark the alignment
        aligned_frames_used[aligned_frame] = True
        mapping_rows.append(aligned_frame)
        mapping_cols.append(n)
    return np.array(mapping_rows, dtype="uint32"), np.array(mapping_cols, dtype="uint32")

def build_cost_matrix_for_pitch(onsets_pred_pitch, df_subset, n_frames, n_neighbours, gaussian_weighting):
    n_notes = len(df_subset)
    cost_matrix = np.zeros((n_frames, n_notes))
    note_idx_map = {}

    for note_idx, (df_idx, row) in enumerate(df_subset.iterrows()):
        frame_idx = int(row.onset_frame)
        frame_start = max(0, frame_idx - n_neighbours)
        frame_end = min(n_frames, frame_idx + n_neighbours + 1)

        if gaussian_weighting:
            window_func = get_gaussian_window_around_note(frame_idx, n_frames, n_neighbours)
        else:
            window_func = 1

        cost_matrix[frame_start:frame_end, note_idx] = (
            onsets_pred_pitch[frame_start:frame_end] * window_func
        ) + EPSILON

        note_idx_map[note_idx] = df_idx

    return cost_matrix, note_idx_map

def build_cost_matrix_for_pitch_vectorized(onsets_pred_pitch, df_subset, n_frames, n_neighbours, gaussian_weighting):
    n_notes = len(df_subset)
    cost_matrix = np.zeros((n_frames, n_notes))
    
    # Vectorize all note parameters
    onset_frames = df_subset['onset_frame'].astype(int).values
    df_indices = df_subset.index.values
    note_idx_map = {note_idx: df_idx for note_idx, df_idx in enumerate(df_indices)}
    
    frame_starts = np.maximum(0, onset_frames - n_neighbours)
    frame_ends = np.minimum(n_frames, onset_frames + n_neighbours + 1)
    
    # Create a mask matrix for efficient assignment
    frame_indices = np.arange(n_frames)[:, None]  # Shape: (n_frames, 1)
    
    # True where frame is within [frame_start, frame_end) for each note
    mask = (frame_indices >= frame_starts) & (frame_indices < frame_ends)
    
    if gaussian_weighting:
        raise ValueError("Cost weghting not yet implemented in vectorized version")
    
    # Vectorized cost calculation
    cost_matrix = (onsets_pred_pitch[:, None] + EPSILON) * mask
    
    return cost_matrix, note_idx_map


def build_cost_matrix_for_pitch_w_frames(onsets_pred_pitch, frame_pred_pitch, df_subset, n_frames,
                                          n_neighbours, gaussian_weighting, max_num_frames = 10, frame_factor = 1):
    """
    Build cost matrix for matching for a single pitch. Include frame information as well for calculating the overall cost
    If established merge with "build_cost_matrix_for_pitch" with a simple switch to enable consideration of frame information
    """
    n_notes = len(df_subset)
    cost_matrix = np.zeros((n_frames, n_notes))
    note_idx_map = {}

    for note_idx, (df_idx, row) in enumerate(df_subset.iterrows()):
        frame_idx = int(row.onset_frame)
        frame_start = max(0, frame_idx - n_neighbours)
        frame_end = min(n_frames, frame_idx + n_neighbours + 1) # non-inclusive

        if gaussian_weighting:
            window_func = get_gaussian_window_around_note(frame_idx, n_frames, n_neighbours)
        else:
            window_func = 1

        num_frames_to_consider = int(min(row.duration_frame, max_num_frames))
        print(num_frames_to_consider)
        frame_based_cost = np.convolve(
        frame_pred_pitch[frame_start:frame_end + num_frames_to_consider -1],
        np.ones(num_frames_to_consider)/ num_frames_to_consider,
        mode='valid')
        # Ensure correct length (handle edge cases)
        frame_length = frame_end - frame_start
        if len(frame_based_cost) < frame_length:
            # Pad if we're near the end of the array
            frame_based_cost = np.pad(frame_based_cost, (0, frame_length - len(frame_based_cost)), 
                                    mode='edge')
        else:
            frame_based_cost = frame_based_cost[:frame_length]


        cost_matrix[frame_start:frame_end, note_idx] = (
            (onsets_pred_pitch[frame_start:frame_end] + frame_factor * frame_based_cost) * window_func
        ) + EPSILON

        note_idx_map[note_idx] = df_idx

    return cost_matrix, note_idx_map


def post_filter_multiplicity_matches(matching, df_subset, onsets_pred_pitch, filter_threshold):
    onset_idx, note_idx = matching
    df_temp = df_subset.reset_index(drop=False)
    df_temp["is_duplicated_onset"] = df_temp.duplicated("onset_frame", keep=False)

    filtered = [[], []]
    for o_idx, n_idx in zip(onset_idx, note_idx):
        if not df_temp.loc[n_idx, "is_duplicated_onset"]:
            filtered[0].append(o_idx)
            filtered[1].append(n_idx)
        else:
            if onsets_pred_pitch[o_idx] >= filter_threshold:
                filtered[0].append(o_idx)
                filtered[1].append(n_idx)
    return filtered

def apply_matching_to_dataframe(df_subset, matching, note_idx_map, matching_score=None):
    df_subset = df_subset.copy()
    onset_idx, note_idx = matching
    for i, (o_idx, n_idx) in enumerate(zip(onset_idx, note_idx)):
        df_idx = note_idx_map[n_idx]
        df_subset.loc[df_idx, "aligned_onset_frame"] = o_idx
        df_subset.loc[df_idx, "matching_score"] = matching_score[i] if matching_score is not None else np.nan
    return df_subset



def match_onsets_from_df(onsets_pred, df_notes, n_neighbours,
                                   frames_pred=None,
                                   gaussian_weighting=False,
                                   pre_filter_multiplicity=True,
                                   post_filter_multiplicity=False,
                                   filter_threshold=0.1,
                                   reorder_alignments_by_measure_position=True,
                                   min_midi=21,
                                   sample_rate=16000,
                                   hop_length=512,
                                   method="bipartite",
                                   return_score=False,
                                   **kwargs):
    """
    Onset matching using bipartite graph matching

    Parameters
    ----------
    onsets_pred : np.ndarray
        A 2-D array of shape '(n_frames, n_pitches)' containing onset activation
        probabilities per frame and pitch.
    df_notes : pandas.DataFrame
        A dataframe of symbolic note events. Must contain at least:
        'start_sec', 'end_sec' or 'duration_sec' and 'pitch' (MIDI pitch).
    n_neighbours : int
        Number of frames (one-sided) for local neighborhood around unaligned onset
    gaussian_weighting : bool, optional
        If True, apply a Gaussian weighting to the cost matrix centered around each unaligned onset
    pre_filter_multiplicity : bool, optional
        If True, remove multiplicity cases *before* matching (i.e., for
        multiple notes with the same onset frame and pitch, keep only one).
        Dropped notes inherit the aligned onset of their kept counterpart after matching.
        Default is False.
    post_filter_multiplicity : bool, optional
        If True, apply multiplicity filtering *after* matching.
        I.e., we first match all onsets including multiplicity onsets and afterwards remove only those
          where the matched onset positions has a onset prediciton below the filter_threshold (see below)
        Ignored if 'pre_filter_multiplicity=True'.  
        Default is False.
    filter_threshold : float, optional
        Threshold used during post-matching multiplicity filtering to decide
        whether to drop ambiguous matches.
        When greedy matching is selected this is used as the "neg_threshold" to decide which onsets to ignore.  Default is 0.1.
    reorder_alignments_by_measure_position : bool, optional
        If True, reorder aligned onsets based on musical measure position
        after alignment to ensure temporal consistency.
        Only makes a difference if we are interested in more than just the onset positions (i.e. instrument labels)
           Default is True.
    min_midi : int, optional
        Lowest MIDI pitch index used to map pitches into 'onsets_pred'.
    method: str, optional
        The alignment method to use, currently "bipartite" and "greedy" are implemented
        Default is "bipartite"
    return_score: bool, optional
        If True, include the matching score for each aligned note in the output dataframe.
    Returns:
    dataframe with columns [start, end, duration, start_initial, end_initial] + whatever columns you had beforehand

    """
    original_index_name = df_notes.index.name
    original_index_col = "__original_index__"
    while original_index_col in df_notes.columns:
        original_index_col = f"_{original_index_col}"

    frame_length = hop_length/sample_rate
    df_notes = prepare_note_dataframe(df_notes, frame_length)
    df_notes[original_index_col] = df_notes.index
    df_notes = df_notes.reset_index(drop=True)

    n_frames = onsets_pred.shape[0]
    if pre_filter_multiplicity:
        assert post_filter_multiplicity == False,\
            "Can't post filter multiplicity matches if we already filtered them out before"
    
    method = method.lower()
    assert method in ["bipartite", "greedy"]

    if pre_filter_multiplicity:
        """
        Of all note events with the same unaligned onset frame, only keep one of them for the matching.
         The other ones will just receive the same aligned_onset time/frame after the matching
        """
        df_notes_unfiltered = df_notes.copy()
        df_notes = df_notes.drop_duplicates(subset=["onset_frame", "pitch"])

    for pitch in df_notes.pitch.unique():
        pitch_index = pitch - min_midi
        df_subset = df_notes[df_notes.pitch == pitch]

        if frames_pred is not None:
            cost_matrix, note_idx_map = build_cost_matrix_for_pitch_w_frames(
                onsets_pred[:, pitch_index], frames_pred[:, pitch_index], df_subset, n_frames, n_neighbours, gaussian_weighting, **kwargs)
        else:
            if gaussian_weighting:
                # TODO gaussian weighting not yet implemented for vecotrized version 
                cost_matrix, note_idx_map = build_cost_matrix_for_pitch(
                                onsets_pred[:, pitch_index], df_subset, n_frames, n_neighbours, gaussian_weighting, **kwargs
                            )
            else:
                
                cost_matrix, note_idx_map = build_cost_matrix_for_pitch_vectorized(
                    onsets_pred[:, pitch_index], df_subset, n_frames, n_neighbours, gaussian_weighting, **kwargs
                )
        # cost matrix is of shape (n_frames, n_notes) and contains the cost for matching each unaligned note to each frame
        if method == "bipartite":
            matching = bipartite_match(csr_array(cost_matrix), maximize=True)
        elif method == "greedy":
            matching = greedy_match(cost_matrix, filter_threshold)

        # matching is a tuple of arrays with ([aligned_frames], [note_idxs])

        if post_filter_multiplicity:
            # Post filter muliplcity makes only sense if we are only interested in IA labels, otherwise there is
            # no meaningful way to match filtered onsets to a different one
            matching = post_filter_multiplicity_matches(
                matching, df_subset, onsets_pred[:, pitch_index], filter_threshold
            )
        # get matching score
        if return_score:
            cost_matrix = cost_matrix.numpy()
            matching_score = cost_matrix[matching[0], matching[1]]
        else:
            matching_score = None

        df_subset = apply_matching_to_dataframe(df_subset, matching, note_idx_map, matching_score)
        df_notes.loc[df_subset.index, ["aligned_onset_frame"]] = df_subset["aligned_onset_frame"]
        if return_score:
            df_notes.loc[df_subset.index, ["matching_score"]] = df_subset["matching_score"]
    
    if pre_filter_multiplicity:
        # Merge the aligend df with the old undfiltered df in order to get aligned start times for the dropped multiplicity notes
        columns = ["onset_frame", "pitch", "aligned_onset_frame"]
        columns.append("matching_score") if return_score else None
        df_merged  = df_notes_unfiltered.merge(
            df_notes[columns],
            on=["onset_frame", "pitch"],
            how="left"
        )
        df_notes = df_merged

    df_notes = finalize_aligned_dataframe(df_notes, frame_length)

    if reorder_alignments_by_measure_position:
        df_notes = reorder_aligned(df_notes)

    df_notes = df_notes.set_index(original_index_col, drop=True)
    df_notes.index.name = original_index_name

    return df_notes


def get_gaussian_window_around_note(frame_idx, n_frames, n_neighbours):
    # get a gaussian window centered around frame_idx
    frame_start = int(max(0, frame_idx - n_neighbours))
    frame_end = int(min(n_frames, frame_idx + n_neighbours + 1))
    win_length = 2 * n_neighbours + 1
    std = n_neighbours
    window_func = scipy.signal.windows.gaussian(win_length, std, sym=True)
    # cut window at edges if necessary

    # Figure out how much to cut from left and right
    cut_left = max(0, n_neighbours - frame_idx)
    cut_right = max(0, (frame_idx + n_neighbours + 1) - n_frames)
    window_func = window_func[cut_left:win_length-cut_right]
    assert window_func.shape[0] == frame_end - frame_start,f"Frame idx {frame_idx}, n_neigbours {n_neighbours}, n_frames {n_frames} Frame start {frame_start}, frame_end {frame_end}, win_length {win_length}, window shape {window_func.shape}"
    return window_func



# For each pitch group: reassign aligned columns ["start_aligned", "end_aligned"]
# in sorted order in increasing measure positions
def reorder_group(group):
    # Sort the aligned values by start_aligned
    aligned_sorted = group[["start_sec", "end_sec"]].sort_values("start_sec").reset_index(drop=True)

    # Assign them back according to start_meas order
    group = group.sort_values("start_meas").reset_index(drop=True)
    group[["start_sec", "end_sec"]] = aligned_sorted
    return group



def reorder_aligned(df):
    # This only makes sense when snapping multi-instrument labels with a IA transcriber
    # Reorder the aligned start and end values, such that - within each pitch -
    # the aligned start and end times are monotonically increasing with measure positions
    # This builds upon the assumption, that notes are played in the right "order" according to their measure positions
    # This assumption might not always be true, but it's our best guess if we only have an IA transcriber
    df_sorted = df.sort_values(["pitch", "start_meas"]).copy()
    df_out = df_sorted.groupby("pitch", group_keys=False).apply(reorder_group)
    df_out.sort_values(by=["start_meas", "part"], inplace=True)
    return df_out


