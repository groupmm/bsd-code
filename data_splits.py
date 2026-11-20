#%%
import os



def validate_filename_BSED(filename):
    assert len(filename.split("_")) == 5, f"{filename} not a valid BSED filename"

def validate_filename_BSD(filename):
    assert len(filename.split("_")) == 3, f"{filename} not a valid BSD filename"

def get_op_from_file_BSED(filename):
    validate_filename_BSED(filename)
    # Return Symphony Op Nr. e.g. "Op021"
    filename = os.path.splitext(filename)[0]
    return filename.split("_")[3].split("-")[0]

def get_mvmtID_from_file_BSED(filename):
    # Return the Op + mvmtm_number, i.e., a uniquely identifiable mvmt, e.g. "Op021-01"
    validate_filename_BSED(filename)
    filename = os.path.splitext(filename)[0]
    return filename.split("_")[3]

def get_mvmtNr_from_file_BSED(filename):
    # Returns only the mvmt number as a string ranging from "01" to "05"
    validate_filename_BSED(filename)
    filename = os.path.splitext(filename)[0]
    mvmt = get_mvmtID_from_file_BSED(filename)
    return mvmt.split("-")[1]

def get_versionID_from_file_BSED(filename):
    validate_filename_BSED(filename)
    filename = os.path.splitext(filename)[0]
    return filename.split("_")[4]

def get_BSEDID_from_file_BSED(filename):
    validate_filename_BSED(filename)
    filename = os.path.splitext(filename)[0]
    return filename.split("_")[0]

def get_splitID_from_file_BSED(filename):
    validate_filename_BSED(filename)
    filename = os.path.splitext(filename)[0]
    return filename.split("_")[1]


# BSD
def get_op_from_file_BSD(filename):
    # Return Symphony Op Nr. e.g. "Op021"
    validate_filename_BSD(filename)
    filename = os.path.splitext(filename)[0]
    return filename.split("_")[1].split("-")[0]

def get_mvmtID_from_file_BSD(filename):
    # Return the Op + mvmtm_number, i.e., a uniquely identifiable mvmt, e.g. Op021-01
    validate_filename_BSD(filename)
    filename = os.path.splitext(filename)[0]
    return filename.split("_")[1]

def get_mvmtNr_from_file_BSD(filename):
    # Returns only the mvmt number as a string ranging from "01" to "05"
    validate_filename_BSD(filename)
    filename = os.path.splitext(filename)[0]
    mvmt = get_mvmtID_from_file_BSD(filename)
    return mvmt.split("-")[1]

def get_versionID_from_file_BSD(filename):
    validate_filename_BSD(filename)
    filename = os.path.splitext(filename)[0]
    return filename.split("_")[2]



def get_movement_split(bsed_files, bsd_files, ignore_Op12504 = True):
    """
    Parameters
    ----------
    bsed_files : list of str
        List of file names or paths belonging to the BSED
    bsd_files : list of str
        List of file names or paths belonging to the BSD
    ignore_Op12504 : bool, optional
        If True, excludes movement Op. 125-04 (fourth movement of Symphony No. 9) from the training set.
        Default is True.

    Returns
    -------
    training_files : list of str
        BSD files that remain in the training set after data split
    bsed_files : list of str
        BSED files
    """
    bsed_mvmts = [get_mvmtID_from_file_BSED(f) for f in bsed_files]
    training_files = [f for f in bsd_files if get_mvmtID_from_file_BSD(f) not in bsed_mvmts]
    if ignore_Op12504:
        training_files = [f for f in training_files if "Op125-04" not in f]
    return training_files, bsed_files

def get_version_split_whole_symphony(bsed_files, bsd_files, ignore_Op12504 = True):
    """
    Parameters
    ----------
    bsed_files : list of str
        List of file names or paths belonging to the BSED
    bsd_files : list of str
        List of file names or paths belonging to the BSD
    ignore_Op12504 : bool, optional
        If True, excludes movement Op. 125-04 (fourth movement of Symphony No. 9) from the training set.
        Default is True.

    Returns
    -------
    training_files : list of str
        BSD files that remain in the training set after data split
    bsed_files : list of str
        BSED files
    """
    bsed_audio_versions = [get_versionID_from_file_BSED(f) for f in bsed_files]
    training_files = [f for f in bsd_files if get_versionID_from_file_BSD(f) not in bsed_audio_versions]
    if ignore_Op12504:
        training_files = [f for f in training_files if "Op125-04" not in f]
    return training_files, bsed_files

def get_version_split_single_movement(bsed_files, bsd_files, ignore_Op12504 = True):
    """
    Parameters
    ----------
    bsed_files : list of str
        List of file names or paths belonging to the BSED
    bsd_files : list of str
        List of file names or paths belonging to the BSD
    ignore_Op12504 : bool, optional
        If True, excludes movement Op. 125-04 (fourth movement of Symphony No. 9) from the training set.
        Default is True.

    Returns
    -------
    training_files : list of str
        BSD files that remain in the training set after data split
    bsed_files : list of str
        BSED files
    """
    bsed_mvmt_audio_versions = [get_mvmtID_from_file_BSED(f) + "_" + get_versionID_from_file_BSED(f) for f in bsed_files]
    training_files = [f for f in bsd_files if (
            get_mvmtID_from_file_BSD(f) + "_" + get_versionID_from_file_BSD(f)) not in bsed_mvmt_audio_versions]
    if ignore_Op12504:
        training_files = [f for f in training_files if "Op125-04" not in f]
    return training_files, bsed_files

def get_neither_split(bsed_files, bsd_files, ignore_Op12504 = True):
    """
    Parameters
    ----------
    bsed_files : list of str
        List of file names or paths belonging to the BSED
    bsd_files : list of str
        List of file names or paths belonging to the BSD
    ignore_Op12504 : bool, optional
        If True, excludes movement Op. 125-04 (fourth movement of Symphony No. 9) from the training set.
        Default is True.

    Returns
    -------
    training_files : list of str
        BSD files that remain in the training set after data split
    bsed_files : list of str
        BSED files
    """
    training_files, bsed_files = get_version_split_whole_symphony(bsed_files, bsd_files, ignore_Op12504)
    training_files, bsed_files = get_movement_split(bsed_files, training_files, ignore_Op12504)
    return training_files, bsed_files

def get_recording_split_reduced_BSED(bsed_files, bsd_files, ignore_Op12504 = True):
    """
    Parameters
    ----------
    bsed_files : list of str
        List of file names or paths belonging to the BSED
    bsd_files : list of str
        List of file names or paths belonging to the BSD
    ignore_Op12504 : bool, optional
        If True, excludes movement Op. 125-04 (fourth movement of Symphony No. 9) from the training set.
        Default is True.

    Returns
    -------
    training_files : list of str
        BSD files that remain in the training set after data split
    bsed_files : list of str
        BSED files
    """
    training_files = bsd_files # No recording overlap between "reduced" BSED and BSD
    bsed_files = [f for f in bsed_files if "Blomstedt" in f or "Synth" in f]
    if ignore_Op12504:
        training_files = [f for f in training_files if "Op125-04" not in f]
    return training_files, bsed_files

