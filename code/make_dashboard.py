"""
make_dashboard.py
One HTML file for ALL participants: raw vs. cleaned pulse and breathing, the rates derived from them,
heartbeat-interval checks (Poincare plots, histogram), and a keep / exclude rating + note for each
participant that is kept in the browser and saved to ONE csv file.
At the top, one figure for the whole group: the response to task events (positive control), to standard
events (comparison) and to task events under the other alignment rule.

Blinding: every participant is shown under a letter (A, B, C, ...) dealt at random, so the reviewer cannot tell who is
who while rating. The key is written to blinding_key.csv. Leave it closed until the ratings are final, then
run unblind_ratings.py.

Run:  python make_dashboard.py        (writes physio_qc_dashboard.html next to the data folder)
"""
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
import process_physio as pp

OUT_HTML = pp.ROOT / "physio_qc_dashboard.html"
KEY_FILE = pp.ROOT / "blinding_key.csv"
RATINGS_FILE = pp.ROOT / "physio_qc_ratings.csv"   # optional. Put the ratings saved from the dashboard here and build again:
                                                   # participants rated "exclude" are then left out of the group figure.
GROUP_MIN_QUALITY = 0.9                            # until a participant is rated, a pulse recording with a PPG quality index
                                                   # below this is left out of the heart-rate panel of the group figure.
ZOOM_PPG_S, ZOOM_RESP_S = 20, 60
GREY, BLUE, ORANGE, PURPLE = "#9a9a9a", "#0072B2", "#E69F00", "#AA3377"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left"})
TIME_S = np.arange(int(round(pp.EPOCH_S[0] * pp.RATE_HZ)), int(round(pp.EPOCH_S[1] * pp.RATE_HZ))) / pp.RATE_HZ
GROUP = [("heart_rate", "heart_rate", "Heart rate", "beats / min", BLUE, pp.HR_WINDOW_S),          # signal, ratings column, name, unit, colour, window
         ("breathing_rate", "breathing", "Breathing rate", "breaths / min", PURPLE, pp.BR_WINDOW_S)]


def letter(n):
    """0 -> A, 1 -> B, ... 25 -> Z, 26 -> AA, 27 -> AB, ..."""
    name, n = "", n + 1
    while n:
        n, r = divmod(n - 1, 26)
        name = chr(65 + r) + name
    return name


def blind_codes(subs):
    """A letter for every participant, handed out in random order, so the reviewer cannot tell who is who.
    The letters are stored in blinding_key.csv and reused, so they never change when the dashboard is rebuilt
    (delete that file and they are dealt again). No fixed seed: the key cannot be regenerated from this script.
    Returns {participant: letter} in the order A, B, C, ..."""
    key = pd.read_csv(KEY_FILE) if KEY_FILE.exists() else pd.DataFrame(columns=["code", "participant"])
    new = [sub for sub in subs if sub not in set(key["participant"])]
    np.random.default_rng().shuffle(new)
    for sub in new:
        key.loc[len(key)] = [letter(len(key)), sub]
    key.to_csv(KEY_FILE, index=False)
    return dict(zip(key["participant"], key["code"]))


def figure(p):
    """One picture per participant. Returns it as text that can be embedded in the HTML."""
    ppg_raw, resp_raw = p["ppg_raw"] - np.median(p["ppg_raw"]), p["resp_raw"] - np.median(p["resp_raw"])
    c, r = p["ppg"], p["resp"]
    t_c, t_r = np.arange(len(ppg_raw)) / pp.PPG_HZ, np.arange(len(resp_raw)) / pp.RESP_HZ

    # three 20 s stretches to enlarge: lowest, median and highest NeuroKit quality (breathing: 60 s around the same moments)
    n = ZOOM_PPG_S * pp.PPG_HZ
    window_quality = c["quality"][: len(t_c) // n * n].reshape(-1, n).mean(axis=1)
    order = np.argsort(window_quality)
    zooms = [("lowest", order[0] * ZOOM_PPG_S), ("median", order[len(order) // 2] * ZOOM_PPG_S), ("highest", order[-1] * ZOOM_PPG_S)]
    resp_zooms = [float(np.clip(start + ZOOM_PPG_S / 2 - ZOOM_RESP_S / 2, 0, pp.KEEP_S - ZOOM_RESP_S)) for _, start in zooms]

    fig = plt.figure(figsize=(12, 10.5), layout="constrained")
    rows = fig.add_gridspec(5, 1, height_ratios=[1, 1.4, 1, 1.4, 2.1])
    fig.legend(loc="outside upper center", ncols=4, frameon=False, handles=[
        Line2D([], [], color=GREY, lw=2.5, label="raw recording"),
        Line2D([], [], color=ORANGE, lw=2.5, marker="o", mfc="none", label="heartbeats as first detected"),
        Line2D([], [], color=BLUE, lw=2.5, marker="o", label="pulse and heartbeats after cleaning and correction"),
        Line2D([], [], color=PURPLE, lw=2.5, label="breathing after cleaning")])

    ax = fig.add_subplot(rows[0])                                      # heart rate, whole run
    ax.plot(t_c, c["rate_found"], color=ORANGE, lw=1.0)
    ax.plot(t_c, c["rate"], color=BLUE, lw=1.0)
    for _, start in zooms:
        ax.axvspan(start, start + ZOOM_PPG_S, color="k", alpha=0.12, lw=0)
    ax.set(title="Heart rate, whole run. Shaded: the three stretches enlarged below", ylabel="beats / min", ylim=(40, 140), xlim=(0, pp.KEEP_S))

    cols = rows[1].subgridspec(1, 3)                                   # pulse, enlarged
    for k, (name, start) in enumerate(zooms):
        ax = fig.add_subplot(cols[k])
        w = slice(int(start * pp.PPG_HZ), int((start + ZOOM_PPG_S) * pp.PPG_HZ))
        ax.plot(t_c[w], ppg_raw[w], color=GREY, lw=1.0)
        ax.plot(t_c[w], c["clean"][w], color=BLUE, lw=1.0)
        for peaks, style in [(c["peaks_found"], dict(marker="o", mfc="none", mec=ORANGE, ms=8, mew=1.5)),
                             (c["peaks"], dict(marker="o", mfc=BLUE, mec=BLUE, ms=3.5))]:
            inside = peaks[(peaks >= w.start) & (peaks < w.stop)]
            ax.plot(t_c[inside], c["clean"][inside], ls="none", **style)
        ax.set(title=f"Pulse, 20 s with {name} quality", xlim=(start, start + ZOOM_PPG_S), ylabel="PPG (a.u.)" if k == 0 else "")

    ax = fig.add_subplot(rows[2])                                      # breathing rate, whole run
    ax.plot(t_r, r["rate"], color=PURPLE, lw=1.0)
    for start in resp_zooms:
        ax.axvspan(start, start + ZOOM_RESP_S, color="k", alpha=0.12, lw=0)
    ax.set(title="Breathing rate, whole run. Shaded: the three stretches enlarged below", ylabel="breaths / min", ylim=(0, 40),
           xlim=(0, pp.KEEP_S), xlabel="time from first kept fMRI volume (s)")

    cols = rows[3].subgridspec(1, 3)                                   # breathing, enlarged
    for k, start in enumerate(resp_zooms):
        ax = fig.add_subplot(cols[k])
        w = slice(int(start * pp.RESP_HZ), int((start + ZOOM_RESP_S) * pp.RESP_HZ))
        ax.plot(t_r[w], resp_raw[w], color=GREY, lw=1.0)
        ax.plot(t_r[w], r["clean"][w], color=PURPLE, lw=1.0)
        for marks, marker in [(r["peaks"], "^"), (r["troughs"], "v")]:
            inside = marks[(marks >= w.start) & (marks < w.stop)]
            ax.plot(t_r[inside], r["clean"][inside], ls="none", marker=marker, color=PURPLE, ms=5)
        ax.set(title="Breathing, 60 s around the same moment", xlim=(start, start + ZOOM_RESP_S), ylabel="belt (a.u.)" if k == 0 else "")

    # heartbeat intervals. Axes run from 1/4 to 2.5 times this person's median interval, so a skipped beat
    # (2 x median) or a doubled beat (1/2 x median) lands in the same place on everybody's plot.
    cols = rows[4].subgridspec(1, 3, width_ratios=[1, 1, 2.2])
    rr_found, rr = np.diff(c["peaks_found"]) / pp.PPG_HZ, np.diff(c["peaks"]) / pp.PPG_HZ
    median = np.median(rr)
    lim, marks = (0.25 * median, 2.5 * median), [0.5 * median, median, 2 * median]
    names = [f"{marks[0]:.2f}\nhalf", f"{marks[1]:.2f}\nmedian", f"{marks[2]:.2f}\ndouble"]
    for k, (name, x, color) in enumerate([("as detected", rr_found, ORANGE), ("after correction", rr, BLUE)]):
        ax = fig.add_subplot(cols[k])                                  # Poincare plot: each interval against the next one
        x = np.clip(x, *lim)
        for m in marks:
            ax.axvline(m, color="0.85", lw=0.8)
            ax.axhline(m, color="0.85", lw=0.8)
        ax.plot(x[:-1], x[1:], ls="none", marker="o", ms=2.5, color=color, alpha=0.6)
        ax.set(title=f"Intervals, {name}", xlim=lim, ylim=lim, xticks=marks, yticks=marks, xticklabels=names,
               yticklabels=[f"{m:.2f}" for m in marks], xlabel="interval (s)", ylabel="next interval (s)" if k == 0 else "")
        ax.set_box_aspect(1)

    ax = fig.add_subplot(cols[2])                                      # how many intervals of each length
    bins = np.linspace(lim[0], lim[1], 46)
    for m in marks:
        ax.axvline(m, color="0.85", lw=0.8)
    ax.hist(np.clip(rr_found, *lim), bins=bins, histtype="step", color=ORANGE, lw=2.0)
    ax.hist(np.clip(rr, *lim), bins=bins, histtype="step", color=BLUE, lw=1.0)
    ax.set(title="Interval counts", yscale="log", ylim=(0.7, 2000), xlim=lim, xticks=marks, xticklabels=names,
           xlabel="interval (s)", ylabel="beats")
    ax.set_yticks([1, 10, 100, 1000], labels=["1", "10", "100", "1000"])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def rated(column):
    """What each participant was rated in the saved ratings file: {participant: "keep" or "exclude"}.
    Empty if there is no ratings file yet; participants not rated yet are simply absent."""
    if not RATINGS_FILE.exists():
        return {}
    merged = pd.read_csv(KEY_FILE).merge(pd.read_csv(RATINGS_FILE, dtype=str), left_on="code", right_on="recording")
    return {sub: rating for sub, rating in zip(merged["participant"], merged[column]) if rating in ("keep", "exclude")}


def in_group_figure(sub, signal, ratings, quality):
    """Your rating decides. A participant who is not rated yet is left out of the heart-rate panel if the PPG quality index is low."""
    if sub in ratings:
        return ratings[sub] == "keep"
    return signal != "heart_rate" or quality[sub] >= GROUP_MIN_QUALITY


def group_figure(evoked, evoked_other, quality):
    """Positive control for the whole dataset: the average change in heart rate and breathing rate after targets,
    after standards, and after targets when the recordings are cut with the OTHER alignment rule.
    Lines are means across participants, bands are 95% confidence intervals."""
    other = "start" if pp.ALIGN == "end" else "end"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for ax, (signal, column, name, unit, color, window) in zip(axes, GROUP):
        ratings = rated(column)
        keep = [sub for sub in evoked if in_group_figure(sub, signal, ratings, quality)]
        for label, c, width, source, events in [("standards", "0.45", 1.2, evoked, "standards"),
                                                (f"targets, recording cut from its {other.upper()} (the other rule)", ORANGE, 1.8, evoked_other, "targets"),
                                                (f"targets, recording cut from its {pp.ALIGN.upper()} (the rule in use)", color, 2.4, evoked, "targets")]:
            curves = np.array([source[sub][signal][events] for sub in keep])
            mean = curves.mean(axis=0)
            if len(keep) > 1:
                half = stats.t.ppf(0.975, len(keep) - 1) * curves.std(axis=0, ddof=1) / np.sqrt(len(keep))
                ax.fill_between(TIME_S, mean - half, mean + half, color=c, alpha=0.2, lw=0)
            ax.plot(TIME_S, mean, color=c, lw=width, label=label)
        ax.axhline(0, color="0.6", lw=0.6)
        low, high = ax.get_ylim()
        ax.plot(window, [low] * 2, color="k", lw=5, solid_capstyle="butt")       # when a response is expected
        ax.set_ylim(low, low + 1.5 * (high - low))                               # head room for the key
        ax.set(title=f"{name}, {len(keep)} of {len(evoked)} participants", xlim=pp.EPOCH_S, xlabel="time from event (s)", ylabel=f"change ({unit})")
        ax.legend(loc="upper right", frameon=False, fontsize=7)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


SECTION = """
<section id="{sub}">
  <h2>Participant {sub}</h2>
  <p class="numbers">PPG quality index {ppg_quality:.3f} &nbsp;|&nbsp; {beats} beats, {beats_changed_by_correction} positions changed by correction
     &nbsp;|&nbsp; {pct_implausible_intervals:.1f}% of beat-to-beat intervals outside 0.4-2.0 s
     &nbsp;|&nbsp; mean {heart_rate_bpm:.0f} beats/min, {breathing_rate_bpm:.0f} breaths/min</p>
  <img src="data:image/png;base64,{png}" alt="{sub}">
  <div class="rate">
    <span><b>Heart rate:</b>
      <label><input type="radio" name="{sub}_cardiac" value="keep"> keep</label>
      <label><input type="radio" name="{sub}_cardiac" value="exclude"> exclude</label></span>
    <span><b>Breathing:</b>
      <label><input type="radio" name="{sub}_breathing" value="keep"> keep</label>
      <label><input type="radio" name="{sub}_breathing" value="exclude"> exclude</label></span>
    <textarea name="{sub}_note" rows="1" placeholder="notes"></textarea>
    <button class="small" onclick="clearRatings('{sub}')">clear ratings</button>
  </div>
</section>"""


GROUP_FIGURE = """<img src="data:image/png;base64,{png}" alt="group task response" style="max-width:1000px">
<p class="numbers">Lines are means across participants; bands are 95% confidence intervals. The black bar marks when a response is expected.
Left out: participants rated “exclude” and, among those not rated yet, pulse recordings with a PPG quality index below {min_quality}.</p>"""


# The text at the top of the dashboard. Edit freely. __GROUP_FIGURE__ marks where the group figure goes;
# it is filled in when the dashboard is built (and left empty when USE_EVENTS is False).
INTRO = """
<h3>What is this for?</h3>
<p>If you have collected physiological data along with your functional MRI data, you may want to use it for denoising (e.g., RETROICOR).
Standard MRI pipelines usually include visual quality control dashboards
to assess your data for alignment, motion, or distortion problems. However, most pipelines do not offer the same type of visual quality control
for physiological data. This dashboard provides that so you can check the data that go into your denoising.</p>
<p>To demonstrate the utility of this dashboard and illustrate the range of possibilities, we have curated a set of participants that includes
equal numbers with very poor-quality physiological data, median quality data, and very high quality data.</p>

<h3>Scoring blind</h3>
<p>To avoid unintentional bias, participant identities are replaced with letters assigned at random. The key is in <i>blinding_key.csv</i>;
do not open it until your ratings are final. When you’re done, run <i>unblind_ratings.py</i> to match the letters with your real participant IDs.</p>
<p>Choose keep or exclude for each signal and add notes if useful. <b>Save ratings to CSV</b> writes everything to one file;
<b>Load ratings from CSV</b> brings a saved file back.</p>

<h3>What is shown</h3>
<p>This dashboard shows, in order:</p>
<ul>
<li>Instantaneous heart rate, before (gold) and after (blue) correction with NeuroKit2.</li>
<li>Raw PPG recordings (gray), with cleaned (blue) traces and beat detection. Gold circles are beats as first detected; blue dots are beats after correction.
A gold circle without a blue dot is a beat that correction removed or moved, and a blue dot without a gold circle is a beat that correction added.
Three sections are shown for each person: the lowest quality 20 s, median quality 20 s, and highest quality 20 s.</li>
<li>Instantaneous respiratory rate.</li>
<li>Raw pneumatic belt recordings (gray), with cleaned (purple) traces. Three sections are shown for each person, matching the three sections
used for the PPG recordings, since these may align with moments of high motion.</li>
<li>Poincaré plots showing inter-beat intervals before (gold) and after (blue) correction with NeuroKit2.</li>
<li>Histogram of inter-beat interval length before (gold) and after (blue) correction with NeuroKit2.</li>
<li>Optionally, above the first participant, one figure for the whole group showing the average change in heart rate and respiratory rate after task events.
This requires additional inputs identifying event times. Until ratings exist, pulse recordings with a low PPG quality index are left out of the heart-rate panel.
Once the saved ratings file is placed in the package folder and the dashboard is built again, participants rated “exclude” are left out instead.</li>
</ul>

<h3>What to look for</h3>
<ul>
<li>How much does the heart rate appear to vary after cleaning and correction? Note extremes: L runs off both ends of the 40-140 axis and is clearly
non-physical, while K varies only modestly around 80 bpm. Another participant, O, has higher variability than K but that data does not look artifactual
and probably demonstrates healthy respiratory sinus arrhythmia.</li>
<li>Are heartbeats retained (blue dots within gold circles) on signal that appears to be noise (e.g., participant A)?</li>
<li>Are heartbeats retained that have approximately double or half the median length (e.g., half in participant D, double in participant F)? How many?
This occurs when peaks are missed, or when the smaller second wave of the pulse (the diastolic peak, which follows the dicrotic notch) is mistakenly
identified as a beat, respectively, and should be absent after cleaning and correction.</li>
<li>How much does respiratory rate vary? There are no obviously poor or unusable recordings in this dataset.</li>
<li>Are breaths retained on segments of the signal that are not clearly complete, independent breaths (e.g., participant D)? Is this common?</li>
<li>If you have included event data: is the expected physiological response to events present in the group figure? If not, alignment between
the physiological data and the functional or task data may be incorrect.</li>
</ul>
__GROUP_FIGURE__
"""

PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Physio QC</title>
<style>
 body{font-family:Helvetica,Arial,sans-serif;margin:0;color:#222}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #bbb;padding:8px 20px;display:flex;gap:16px;align-items:center;z-index:5}
 header b{font-size:17px} button,.filebutton{font-size:14px;padding:5px 12px;border:1px solid #555;border-radius:4px;background:#f3f3f3;cursor:pointer}
 button.small{font-size:12px;padding:2px 8px} #status{font-size:13px;color:#0a5d2a}
 main{padding:0 20px 60px;max-width:1240px} section{border-top:2px solid #ddd;margin-top:26px;padding-top:6px}
 h2{margin:6px 0 2px} .numbers{margin:6px 0 4px;font-size:13px;color:#444} img{width:100%;height:auto}
 .rate{display:flex;gap:22px;align-items:center;background:#f4f6f8;padding:8px 10px;font-size:15px}
 .rate label{margin-left:8px} textarea{flex:1;font-size:14px;font-family:inherit}
 .how{font-size:14px;line-height:1.45;max-width:980px} .how h3{margin:18px 0 4px;font-size:16px} .how ul{margin:4px 0 8px;padding-left:22px} .how li{margin-bottom:4px} #jump a{margin-right:9px;font-size:14px} a.done{color:#888;text-decoration:line-through}
</style></head><body>
<header><b>Physio QC</b><span id="progress"></span>
 <button onclick="saveCSV()">Save ratings to CSV</button>
 <label class="filebutton">Load ratings from CSV<input type="file" accept=".csv" style="display:none" onchange="loadCSV(this.files[0]); this.value=''"></label>
 <span id="status"></span>
</header>
<main>
<div class="how">
__INTRO__
<p id="jump"></p>
</div>
__SECTIONS__
</main>
<script>
const RATINGS = ['cardiac', 'breathing'];
const recordings = [...document.querySelectorAll('section')].map(s => s.id);
const KEY = 'physio_qc_ratings:' + location.pathname;      // one memory per dashboard file

// Ratings are kept in the browser's own storage. If the browser forbids that, they last only until the page is closed.
let memory = {}, canStore = true;
try { localStorage.getItem(KEY); } catch (e) { canStore = false; }
function read(){ return canStore ? JSON.parse(localStorage.getItem(KEY) || '{}') : memory; }
function write(d){ if (canStore) localStorage.setItem(KEY, JSON.stringify(d)); else memory = d; }
function say(text){ document.getElementById('status').textContent = text; }
function rated(d, r){ return RATINGS.every(f => d[r + '_' + f]); }

function remember(event){                 // runs on every click and keystroke
  if (event && event.target && event.target.type === 'file') return;
  const d = read();
  document.querySelectorAll('input[type=radio]').forEach(e => { if (e.checked) d[e.name] = e.value; });
  document.querySelectorAll('textarea').forEach(e => { if (e.value) d[e.name] = e.value; else delete d[e.name]; });
  write(d); refresh();
}

function refresh(){                       // update the counter and the list of participants
  const d = read();
  document.getElementById('jump').innerHTML = recordings.map(r => `<a href="#${r}" class="${rated(d, r) ? 'done' : ''}">${r}</a>`).join('');
  document.getElementById('progress').textContent = recordings.filter(r => rated(d, r)).length + ' of ' + recordings.length + ' rated';
}

function show(){                          // put the stored ratings back on the page
  const d = read();
  document.querySelectorAll('input[type=radio]').forEach(e => { e.checked = d[e.name] === e.value; });
  document.querySelectorAll('textarea').forEach(e => { e.value = d[e.name] || ''; });
  refresh();
}

function clearRatings(r){ const d = read(); for (const f of RATINGS) delete d[r + '_' + f]; write(d); show(); }

// ---- one CSV file holds everything
const COLUMNS = ['recording', 'heart_rate', 'breathing', 'note'];
const FIELDS = ['_cardiac', '_breathing', '_note'];
function quote(v){ return '"' + String(v || '').replace(/"/g, '""') + '"'; }
function toCSV(){
  const d = read();
  const rows = recordings.map(r => [r].concat(FIELDS.map(f => f === '_note' ? quote(d[r + f]) : (d[r + f] || ''))).join(','));
  return [COLUMNS.join(',')].concat(rows).join('\n') + '\n';
}
function saveCSV(){
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([toCSV()], {type: 'text/csv'}));
  link.download = 'physio_qc_ratings.csv';
  document.body.appendChild(link); link.click(); link.remove();
  const d = read();
  say('Saved physio_qc_ratings.csv at ' + new Date().toLocaleTimeString() + ': ' + recordings.length + ' participants, '
      + recordings.filter(r => rated(d, r)).length + " fully rated. Look in your browser's download folder.");
}
function splitCSV(text){                  // copes with commas, quotes and line breaks inside a note
  const rows = [[]]; let cell = '', quoted = false;
  for (let i = 0; i < text.length; i++){
    const ch = text[i];
    if (quoted){ if (ch === '"' && text[i + 1] === '"'){ cell += '"'; i++; } else if (ch === '"') quoted = false; else cell += ch; }
    else if (ch === '"') quoted = true;
    else if (ch === ','){ rows[rows.length - 1].push(cell); cell = ''; }
    else if (ch === '\n'){ rows[rows.length - 1].push(cell); cell = ''; rows.push([]); }
    else if (ch !== '\r') cell += ch;
  }
  if (cell || rows[rows.length - 1].length) rows[rows.length - 1].push(cell);
  return rows.filter(r => r.length > 1);
}
function applyCSV(text, name){
  const d = {}; let n = 0;
  for (const row of splitCSV(text).slice(1)){
    const r = row[0]; if (!recordings.includes(r)) continue; n++;
    FIELDS.forEach((f, i) => { if (row[i + 1]) d[r + f] = row[i + 1]; });
  }
  write(d); show(); say('Loaded ratings for ' + n + ' participants from ' + name + '.');
}
function loadCSV(file){ if (file) file.text().then(text => applyCSV(text, file.name)); }

document.addEventListener('change', remember); document.addEventListener('input', remember);
show();
if (!canStore) say('This browser is not keeping ratings between visits. Save the CSV before closing the page.');
</script></body></html>"""


def build():
    codes = blind_codes(pp.subjects())
    other_rule = "start" if pp.ALIGN == "end" else "end"
    sections, evoked, evoked_other, quality = [], {}, {}, {}
    for sub in [sub for sub in codes if sub in pp.subjects()]:      # in the order A, B, C, ... = random order of participants
        p = pp.process_participant(sub)
        if p is None:                 # unexpected recording length: nothing is drawn
            continue
        numbers = pp.summarize(p)
        numbers["sub"] = codes[sub]   # only the letter goes into the HTML
        quality[sub] = numbers["ppg_quality"]
        if pp.USE_EVENTS:             # only for the group figure
            evoked[sub] = p["evoked"]
            evoked_other[sub] = pp.process_participant(sub, other_rule)["evoked"]
        sections.append(SECTION.format(png=figure(p), **numbers))

    group = GROUP_FIGURE.format(png=group_figure(evoked, evoked_other, quality), min_quality=GROUP_MIN_QUALITY) if pp.USE_EVENTS else ""
    page = PAGE.replace("__INTRO__", INTRO).replace("__GROUP_FIGURE__", group).replace("__SECTIONS__", "\n".join(sections))
    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"{len(sections)} participants -> {OUT_HTML}")


if __name__ == "__main__":
    build()
