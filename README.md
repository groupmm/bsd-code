# BSED and BSD: Beethoven Symphony Datasets

A repository accompaniying the Beethoven Symphony Excerpts Dataset (BSED) and the Beethoven Symphony Dataset (BSD). Provides code for score parsing, recreating the BSED audio from the BSD, and score-audio alignment.

Links: [Paper](https://transactions.ismir.net/articles/10.5334/tismir.343), [Dataset on Zenodo](https://zenodo.org/records/20344500), [Demo Page](https://audiolabs-erlangen.de/resources/MIR/2026-BSED-BSD)

If you use the dataset and/or parts of this repository, please cite the following paper:

> Berendes, H.-U., Saha, A., Maman, B., Arifi-Müller, V. and Müller, M. (2026) ‘Beethoven Symphony Excerpt Dataset (BSED): An Evaluation Dataset for Orchestral Music Transcription’, Transactions of the International Society for Music Information Retrieval, 9(1), p. 405–422. Available at: https://doi.org/10.5334/tismir.343.

```bibtex
@article{BerendesSMAM26_BSED,
author = {Berendes, Hans-Ulrich and Saha, Abhirup and Maman, Ben and Arifi-Müller, Vlora and Müller, Meinard},
title = {Beethoven Symphony Excerpt Dataset ({BSED}): An Evaluation Dataset for Orchestral Music Transcription},
journal = {Transactions of the International Society for Music Information Retrieval},
volume = {9},
number = {1},
pages = {405--422},
year = {2026},
doi = {10.5334/tismir.343},
}

```



**Important notes for using the BSD (the larger trainset)**:

*Tuning and Transpositions*:
Some recordings in the BSD have tuning deviations larger than a semitone, which in the transcription context creates a mismatch between annotations and audio. In our experiments we always use a "tuning-corrected" pitch-shifted version of the audios. Due to storage constraints we only redistribute the original recordings. For obtaining the pitch-shifted version please run `02_tuning_corretion_BSD.py`. 
The smaller BSED is distributed with the pitch-shifted audio included.

*BSD Alignment*:
For aligning the BSD as in the original release of the dataset (both DTW and snapping) you need the transcription features for the whole dataset. Because of storage constrants we don't distribute this with the dataset release. If you are interested in realigning the full BSD dataset, please contact us.
For the smaller BSED we distribute the transcription features within the dataset.

### Core Scripts

- **`01_parse_BSED_xmls.py`** - Parse MusicXML files of the BSED
- **`02_tuning_correction_BSD.py`** - Pitch shift to normalize tuning in BSD audio recordings; requires rubberband on your system
- **`03_cut_BSED_audio.py`** - Cut the BSED audio files from the BSD
- **`04_alignment_dtw_BSD.py`** - Align BSD audio with scores using Dynamic Time Warping
- **`04_alignment_dtw_BSED.py`** - Align BSED audio with scores using Dynamic Time Warping
- **`05_alignment_snapping_BSD.py`** - Refine BSD alignment using frame-level note snapping
- **`05_alignment_snapping_BSED.py`** - Refine BSED alignment using frame-level note snapping


### Utility Modules

- **musicxml_parser/** - MusicXML parsing modules needed by these workflows
- **`note_level_alignment.py`** - Note-level alignment algorithms
- **`utils_bsd.py`** - Utilities for MIDI conversion, dataframe handling, and sonification
- **`data_splits.py`** - Get data splits


### Configuration
Create the conda environment with:
```shell
conda env create -f environment.yml
```

Specify the paths to the data directories for BSD and BSED in `paths.py`:
```python
bsd_dir = "path_to_BSD"
bsed_dir = "path_to_BSED"
```

Modify these paths according to your local data storage setup before running a script, for example:

```shell
python 01_parse_BSED_xmls.py
```

Scripts may overwrite existing outputs; review their settings before running them.

### Note Event CSV - Column Documentation

| Name | Description | Representation |
|---|---|---|
| `start_meas` | Onset (in the score) of the note as a fractional measure number. | Positive float (rounded to 3 decimal places) |
| `end_meas` | Offset (in the score) of the note as a fractional measure number. If the note ends at the very end of a measure, it is notated exclusively, e.g. `5.999` is correct while `6.000` is incorrect. | Positive float (rounded to 3 decimal places) |
| `start_quarter` | Onset (in the score) from the beginning of the score in quarter notes. This provides a symbolic time axis that is independent of measures. | Positive float |
| `duration_quarter` | Duration of the note in quarter notes (slightly redundant to `start_meas` and `end_meas`). | Positive float |
| `time_sig` | Local time signature at the note's position (e.g. `4/4`, `3/4`). | String |
| `pitch` | Pitch number of the note (`-1` for unpitched drum notes). This is the sounding pitch, i.e. the pitch expected to be heard based on the score (after transposition in case of transposing instruments). | Integer in `[-1, 127]` |
| `pitch_name` | Pitch name in scientific notation (e.g. `C4`, `Bb4`). Flats and sharps are indicated with `#` and `b`. | String |
| `part` | Name of the score part the note belongs to (e.g. `Violin 1`). | String |
| `instrument` | Full instrument name (e.g. `Violin`). | String |
| `articulation` | Articulation marking associated with the note (e.g. staccato, accent). | List of strings (comma-separated) |
| `expression` | Expression associated with the note (e.g. fermatas, mordents, trills, turns). | List of strings (comma-separated) |
| `dynamic` | Dynamic marking that holds for this note, e.g. `pp`, `mf`. | String |
| `velocity` | Realization of dynamic into a velocity value | Integer in [1, 127] |
| `tempo_qpm` | Tempo in quarter notes per minute at the note's start position. | Positive float (rounded to 3 decimal places) |
| `start_sec` | Onset time of the note in seconds (can be derived from the score tempo map or aligned to a performance). | Positive float (rounded to 3 decimal places) |
| `end_sec` | Offset time of the note in seconds (can be derived from the score tempo map or aligned to a performance). | Positive float (rounded to 3 decimal places) |
| `duration_sec` | Duration of the note in seconds. | Positive float (rounded to 3 decimal places) |

## Dataset Errors
We welcome feedback on potential annotation errors or inconsistencies. In case you find something, please [submit an Issue in this GitHub repository.](https://github.com/groupmm/bsd-code/issues/new)

## License
The code for this toolbox is published under an [MIT license](LICENSE).


## Acknowledgements

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (MU 2686/15‑1). The authors are with the [International Audio Laboratories Erlangen](https://audiolabs-erlangen.de/), a joint institution of the [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html). The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich‑Alexander Universität (FAU). The hardware is funded by the German Research Foundation (DFG).