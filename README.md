# Physio QC dashboard example

The QC dashboard is a single HTML page for visual quality control of physiological recordings made during fMRI (pulse and respiration belt), with keep / exclude ratings saved to one CSV file.

Example data: 15 participants, 3T GE scanner (pulse collected at 100 Hz, respiration belt at 25 Hz), oddball task.

## Look at it

Open `physio_qc_dashboard.html` in any browser. 

## Rebuild it

Requires Python with neurokit2, numpy, pandas, scipy and matplotlib (made with NeuroKit2 0.2.13).

    cd code
    python make_dashboard.py      # writes physio_qc_dashboard.html
    python process_physio.py      # optional: writes heart rate and breathing rate to derivatives/

## Use it on your own data

Edit the SETTINGS block at the top of `code/process_physio.py`: data folder, file names, sampling rates, scan length,
alignment rule, event names. A few more settings are at the top of `code/make_dashboard.py`.
Set `USE_EVENTS = False` if you have no event files.

## Things to know

- Participants are shown as letters so that you can score your data blind. The key is `blinding_key.csv`. Keep that file! When your ratings are final, run
  `python unblind_ratings.py path/to/physio_qc_ratings.csv` from the `code` folder to put the IDs back.
- The group figure at the top leaves out participants you rated "exclude". To get that, put your saved
  `physio_qc_ratings.csv` in this folder and run `make_dashboard.py` again. Until a participant is rated, a pulse
  recording with a PPG quality index (from NeuroKit2) below 0.9 is left out of the heart-rate panel. That number serves this figure only.

## Files in `data/`

- `sub-XX_ppg_100hz.txt`, `sub-XX_resp_25hz.txt`: the scanner recordings, one number per line.
- `sub-XX_events.tsv`: event onsets in seconds from the start of the scan, and event names.
