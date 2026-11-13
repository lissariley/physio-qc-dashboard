"""
process_physio.py
Pulse (PPG) and respiration-belt recordings from an MRI scanner
  ->  heart rate, breathing rate and breath amplitude, on the clock of the fMRI run.

To use it on your own data, change the SETTINGS block below. Nothing else needs editing.
The settings as shipped describe the example data (GE scanner, PPG at 100 Hz, belt at 25 Hz).

Steps
  1. Load     One recording per file: one number per line, no header, no timestamps.
  2. Align    Cut the recording down to the part that matches the fMRI data (see ALIGN below).
              Time 0 = start of the first fMRI volume that is kept.
  3. Process  NeuroKit2: clean -> find peaks -> correct peaks -> rate (plus amplitude for breathing).
  4. Control  Average heart rate and breathing rate after rare TARGET events and after frequent STANDARD
              events. The dashboard averages these across participants: a response after targets that
              standards do not produce shows that the recordings and the task share a clock.
  5. Save     One table per participant and one summary row per participant.

Run:  python process_physio.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import neurokit2 as nk

ROOT = Path(__file__).resolve().parent.parent

# =====================================================================================================
# SETTINGS: change these for your own data
# =====================================================================================================

# ---- Where the files are
DATA = ROOT / "data"                     # folder with the recordings and event files, e.g. Path("/Users/me/study/physio")
OUT = ROOT / "derivatives"               # folder the results are written to

# ---- How the files are named. {sub} stands for the participant ID.
PPG_FILE = "{sub}_ppg_100hz.txt"         # pulse recording
RESP_FILE = "{sub}_resp_25hz.txt"        # respiration-belt recording
EVENTS_FILE = "{sub}_events.tsv"         # task events: tab-separated, with the columns  onset_s  and  event
PARTICIPANTS = None                      # None = everyone with a pulse file in DATA. Or a list: ["sub-01", "sub-07"]

# ---- Sampling rates of the recordings (samples per second)
PPG_HZ = 100
RESP_HZ = 25

# ---- The fMRI run
TR = 2.5                                 # seconds per volume
N_VOLS = 240                             # volumes acquired
N_DROPPED = 2                            # volumes dropped from the start of the fMRI data

# ---- Alignment rule: which part of the recording is the scan?
#   "end"    the recording STOPS when the scan stops, and started some time before it (GE scanners)
#   "start"  the recording STARTS when the scan starts, and may run on after it
ALIGN = "end"
EXTRA_S = (28, 34)                       # how much longer than the scan a recording should be, in seconds.
                                         # A file outside this range is skipped: nothing is processed or shown.
                                         # Use (0, float("inf")) to accept any recording at least as long as the scan.

# ---- Task events (the positive control). onset_s counts from the start of the scan, before volumes are dropped.
USE_EVENTS = True                        # False = you have no event files; the group task-response figure is left out
TARGET = "target"                        # name, in the event column, of the events that should produce a response
STANDARD = "standard"                    # name of the comparison events
EPOCH_S = (-2, 12)                       # stretch of time averaged around each event, in seconds
HR_WINDOW_S = (2.5, 3.0)                 # when a heart-rate response to targets is expected (prior knowledge; true for THIS task)
BR_WINDOW_S = (4.5, 5.5)                 # when a breathing-rate response to targets is expected (prior knowledge; true for THIS task)

# ---- Other
IBI_OK_S = (0.4, 2.0)                    # plausible time between heartbeats (150 to 30 beats/min)
RATE_HZ = 25                             # heart rate and breathing rate are saved at this rate

# =====================================================================================================
# End of settings
# =====================================================================================================

SCAN_S = TR * N_VOLS                     # length of the scan
KEEP_S = TR * (N_VOLS - N_DROPPED)       # length of the fMRI data that is kept


def subjects():
    """Participant IDs: the PARTICIPANTS list, or everyone whose pulse file matches PPG_FILE."""
    if PARTICIPANTS is not None:
        return list(PARTICIPANTS)
    before, after = PPG_FILE.split("{sub}")
    return sorted(f.name[len(before):len(f.name) - len(after)] for f in DATA.glob(PPG_FILE.replace("{sub}", "*")))


def load_aligned(path, hz, align=None):
    """Load one recording and cut it to the part that matches the kept fMRI volumes.
    Returns None if the file length is not what EXTRA_S expects (then nothing is processed or shown)."""
    align = ALIGN if align is None else align
    x = np.loadtxt(path)
    extra_s = len(x) / hz - SCAN_S
    if not EXTRA_S[0] <= extra_s <= EXTRA_S[1]:
        print(f"skipped {Path(path).name}: {len(x) / hz:.1f} s recorded")
        return None
    n = int(KEEP_S * hz)
    if align == "end":
        return x[-n:]                                  # the last KEEP_S seconds
    return x[int(N_DROPPED * TR * hz):][:n]            # skip the dropped volumes, then KEEP_S seconds


def at_rate_hz(x, hz):
    """Put a trace sampled at hz on the common output clock (RATE_HZ)."""
    return np.interp(np.arange(int(KEEP_S * RATE_HZ)) / RATE_HZ, np.arange(len(x)) / hz, x)


def process_ppg(ppg):
    clean = nk.ppg_clean(ppg, sampling_rate=PPG_HZ)
    found = nk.ppg_peaks(clean, sampling_rate=PPG_HZ)[1]["PPG_Peaks"]                         # as detected
    peaks = nk.ppg_peaks(clean, sampling_rate=PPG_HZ, correct_artifacts=True)[1]["PPG_Peaks"]  # missed/extra beats repaired
    return dict(
        clean=clean, peaks_found=found, peaks=peaks,
        rate_found=nk.signal_rate(found, sampling_rate=PPG_HZ, desired_length=len(ppg)),
        rate=nk.signal_rate(peaks, sampling_rate=PPG_HZ, desired_length=len(ppg)),
        quality=nk.ppg_quality(clean, peaks=found, sampling_rate=PPG_HZ),   # 0-1: how well each beat matches the average beat
    )


def process_resp(resp):
    signals, info = nk.rsp_process(resp, sampling_rate=RESP_HZ)
    return dict(
        clean=signals["RSP_Clean"].values, peaks=info["RSP_Peaks"], troughs=info["RSP_Troughs"],
        rate=signals["RSP_Rate"].values, amplitude=signals["RSP_Amplitude"].values,
    )


def evoked(rate, onsets):
    """Average change in a rate around events, relative to the time before each event."""
    before, after = int(round(-EPOCH_S[0] * RATE_HZ)), int(round(EPOCH_S[1] * RATE_HZ))
    starts = np.round(onsets * RATE_HZ).astype(int)
    epochs = np.array([rate[s - before:s + after] for s in starts if s - before >= 0 and s + after <= len(rate)])
    return (epochs - epochs[:, :before].mean(axis=1, keepdims=True)).mean(axis=0)


def window_mean(curve, window):
    time_s = np.arange(int(round(EPOCH_S[0] * RATE_HZ)), int(round(EPOCH_S[1] * RATE_HZ))) / RATE_HZ
    return curve[(time_s >= window[0]) & (time_s <= window[1])].mean()


def process_participant(sub, align=None):
    """Everything for one participant. Returns None if either recording has an unexpected length."""
    ppg = load_aligned(DATA / PPG_FILE.format(sub=sub), PPG_HZ, align)
    resp = load_aligned(DATA / RESP_FILE.format(sub=sub), RESP_HZ, align)
    if ppg is None or resp is None:
        return None
    p = dict(sub=sub, ppg_raw=ppg, resp_raw=resp, ppg=process_ppg(ppg), resp=process_resp(resp))
    p["heart_rate"] = at_rate_hz(p["ppg"]["rate"], PPG_HZ)
    p["breathing_rate"] = at_rate_hz(p["resp"]["rate"], RESP_HZ)
    p["breath_amplitude"] = at_rate_hz(p["resp"]["amplitude"], RESP_HZ)

    if USE_EVENTS:
        events = pd.read_csv(DATA / EVENTS_FILE.format(sub=sub), sep="\t")
        events["onset_s"] -= N_DROPPED * TR      # put events on the clock of the kept volumes
        targets = events["onset_s"][events["event"] == TARGET].values
        standards = events["onset_s"][events["event"] == STANDARD].values
        p["evoked"] = {signal: dict(targets=evoked(p[signal], targets), standards=evoked(p[signal], standards))
                       for signal in ("heart_rate", "breathing_rate")}
    return p


def summarize(p):
    ibi = np.diff(p["ppg"]["peaks"]) / PPG_HZ
    return dict(
        sub=p["sub"],
        ppg_quality=np.mean(p["ppg"]["quality"]),
        beats=len(p["ppg"]["peaks"]),
        beats_changed_by_correction=len(set(p["ppg"]["peaks_found"]) ^ set(p["ppg"]["peaks"])),
        pct_implausible_intervals=100 * np.mean((ibi < IBI_OK_S[0]) | (ibi > IBI_OK_S[1])),
        heart_rate_bpm=np.mean(p["ppg"]["rate"]),
        breaths=len(p["resp"]["troughs"]),
        breathing_rate_bpm=np.mean(p["resp"]["rate"]),
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rows = []
    for sub in subjects():
        p = process_participant(sub)
        if p is None:
            continue
        pd.DataFrame(dict(
            time_s=np.arange(len(p["breathing_rate"])) / RATE_HZ,
            heart_rate_bpm=p["heart_rate"],
            breathing_rate_bpm=p["breathing_rate"],
            breath_amplitude=p["breath_amplitude"],
        )).round(3).to_csv(OUT / f"{sub}_physio_{RATE_HZ}hz.csv", index=False)
        rows.append(summarize(p))
    pd.DataFrame(rows).round(3).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
