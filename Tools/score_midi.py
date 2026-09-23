# /// script
# requires-python = ">=3.11"
# dependencies = ["mido", "mir_eval", "numpy"]
# ///
"""Score scribe's raw event JSON (`--format json`) against MIDI ground truth.

    uv run Tools/score_midi.py <run dir> --manifest pairs.json [--csv out.csv] [--verbose]

`pairs.json` maps `<relative json path without .json>` to
`{"group": "<label>", "midi": "<path.mid>"}` (or `"midi": [several parts]`);
a value may add `"window": "midi-span"` to score only inside the MIDI's own
time span (a stem longer than its pattern). Scoring is mir_eval.transcription
with onset tolerance 50 ms, pitch tolerance 50 cents, offsets ignored — the
standard onset+pitch note F1 — reported four ways per file:

  exact    reference as written
  octave   reference shifted by the whole-octave offset that scores best
           (sample-pack patches routinely play one or two octaves off the MIDI)
  any      any semitone shift in ±30 that beats octave by more than 0.05
           (a detuned or transposed patch); the shift histogram says how often
  chroma   pitch class only
  onset    onsets only, any pitch

plus the median signed onset error of matched notes, note counts, files with
no output, drum notes, the label the model chose, and the recall of notes
starting at time zero (the leading-ties question, see CLAUDE.md).

Measured 2026-09-23 on ~700 loops from four commercial packs with MIDI; the
numbers and the caveats (chord-stab and gated patches make more notes sound
than the MIDI holds; pure sub-bass sines go unheard) are in the README.
"""
import argparse, csv, json, os, statistics, sys
from collections import defaultdict
import mido, numpy as np
import mir_eval.transcription as T

TOL = 0.05

def midi_notes(path):
    mf = mido.MidiFile(path)
    tempo = 500000
    for tr in mf.tracks:
        for m in tr:
            if m.type == "set_tempo": tempo = m.tempo; break
    spb = tempo / 1e6 / mf.ticks_per_beat
    notes = []
    for tr in mf.tracks:
        t, on = 0, defaultdict(list)
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0: on[m.note].append(t)
            elif m.type in ("note_off", "note_on") and on.get(m.note):
                s = on[m.note].pop(0); notes.append((s * spb, t * spb, m.note))
    return notes

def json_notes(path):
    events = json.load(open(path))
    starts = {e["index"]: e for e in events if e["type"] == "start"}
    notes = []
    for e in events:
        if e["type"] == "end":
            s = starts[e["start_event_index"]]
            if not s.get("is_drum"): notes.append((s["start_time"], e["end_time"], s["pitch"]))
    return notes

def json_drums(path):
    return sum(1 for e in json.load(open(path)) if e["type"] == "start" and e.get("is_drum"))

def json_instrument(path):
    c = defaultdict(int)
    for e in json.load(open(path)):
        if e["type"] == "start" and not e.get("is_drum"): c[e["instrument"]] += 1
    return max(c, key=c.get) if c else "-"

def hz(p): return 440.0 * 2 ** ((np.asarray(p, dtype=float) - 69) / 12)

def f1(ref, est, pitch=True):
    if not ref or not est: return 0.0, 0.0, 0.0
    ri = np.array([[s, max(e, s + 0.01)] for s, e, _ in ref]); ei = np.array([[s, max(e, s + 0.01)] for s, e, _ in est])
    rp = hz([p for *_, p in ref]); ep = hz([p for *_, p in est])
    if not pitch: rp = np.full(len(ref), 440.0); ep = np.full(len(est), 440.0)
    p, r, f, _ = T.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=TOL, pitch_tolerance=50.0, offset_ratio=None)
    return p, r, f

def onset_errors(ref, est):
    if not ref or not est: return []
    ri = np.array([[s, max(e, s + 0.01)] for s, e, _ in ref]); ei = np.array([[s, max(e, s + 0.01)] for s, e, _ in est])
    m = T.match_notes(ri, hz([p for *_, p in ref]), ei, hz([p for *_, p in est]), onset_tolerance=TOL, pitch_tolerance=50.0, offset_ratio=None)
    return [est[j][0] - ref[i][0] for i, j in m]

def score(ref, est):
    row = {}
    row["exact_p"], row["exact_r"], row["exact_f"] = f1(ref, est)
    best, shift = (0, 0, 0), 0
    for k in range(-3, 4):
        r = f1([(s, e, p + 12 * k) for s, e, p in ref], est)
        if r[2] > best[2]: best, shift = r, k
    row["octave_p"], row["octave_r"], row["octave_f"], row["octave_shift"] = *best, shift
    tbest, tshift = best, 12 * shift
    for k in range(-30, 31):
        if k % 12 == 0: continue
        r = f1([(s, e, p + k) for s, e, p in ref], est)
        if r[2] > tbest[2] + 0.05: tbest, tshift = r, k   # a non-octave shift must clearly win
    row["trans_f"], row["trans_shift"] = tbest[2], tshift
    fold = lambda ns: [(s, e, 60 + p % 12) for s, e, p in ns]
    row["chroma_f"] = f1(fold(ref), fold(est))[2]
    row["onset_f"] = f1(ref, est, pitch=False)[2]
    errs = onset_errors([(s, e, p + tshift) for s, e, p in ref], est)
    row["onset_ms"] = 1000 * statistics.median(errs) if errs else float("nan")
    row["ref_n"], row["est_n"] = len(ref, ), len(est)
    shifted = [(s, e, p + tshift) for s, e, p in ref]
    first = [n for n in shifted if n[0] < 0.05]
    row["first_n"] = len(first)
    row["first_hit"] = f1(first, est)[1] * len(first) if first and est else 0.0   # recall × count = notes found
    return row

def manifest_pairs(run, manifest):
    m = json.load(open(manifest)); out = []
    for key, v in sorted(m.items()):
        j = f"{run}/{key}.json"
        mids = v["midi"] if isinstance(v["midi"], list) else [v["midi"]]
        if os.path.exists(j): out.append((v.get("group", "all"), key.split("/")[-1], j, mids, v.get("window")))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--manifest", required=True); ap.add_argument("--csv"); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    rows = []
    for group, name, j, mids, window in manifest_pairs(a.run, a.manifest):
        mids = [m for m in mids if os.path.exists(m)]
        if not mids: print(f"no MIDI for {name}", file=sys.stderr); continue
        ref = [n for m in mids for n in midi_notes(m)]
        est = json_notes(j)
        row_extra = {"est_drums": json_drums(j), "instrument": json_instrument(j)}
        if window == "midi-span":
            lo, hi = min(s for s, *_ in ref) - TOL, max(e for _, e, _ in ref) + TOL
            est = [n for n in est if lo <= n[0] <= hi]
        row = {"group": group, "name": name, **score(ref, est), **row_extra}
        rows.append(row)
        if a.verbose: print(f"{group:12} {name[:44]:44} exact {row['exact_f']:.2f} oct{row['octave_shift']:+d} {row['octave_f']:.2f} any{row['trans_shift']:+d} {row['trans_f']:.2f} chroma {row['chroma_f']:.2f} onset {row['onset_f']:.2f} {row['onset_ms']:+.0f}ms ref {row['ref_n']} est {row['est_n']} drums {row['est_drums']} {row['instrument']}")
    if a.csv:
        with open(a.csv, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'group':12} {'n':>4} {'exact F1':>9} {'octave F1':>10} {'any-shift':>10} {'chroma F1':>10} {'onset F1':>9} {'onset ms':>9} {'est/ref':>8} {'silent':>6} {'drums':>6} {'t=0 rec':>7}  shifts (semitones)")
    for g in sorted(set(r["group"] for r in rows)) + ["ALL"]:
        rs = [r for r in rows if g == "ALL" or r["group"] == g]
        mean = lambda k: statistics.mean(r[k] for r in rs)
        med = statistics.median([r["onset_ms"] for r in rs if r["onset_ms"] == r["onset_ms"]] or [float("nan")])
        shifts = defaultdict(int)
        for r in rs: shifts[r["trans_shift"]] += 1
        silent = sum(1 for r in rs if r["est_n"] == 0)
        drums = sum(r["est_drums"] for r in rs)
        first = sum(r["first_hit"] for r in rs) / max(1, sum(r["first_n"] for r in rs))
        print(f"{g:12} {len(rs):4d} {mean('exact_f'):9.3f} {mean('octave_f'):10.3f} {mean('trans_f'):10.3f} {mean('chroma_f'):10.3f} {mean('onset_f'):9.3f} {med:+9.0f} {sum(r['est_n'] for r in rs)/max(1,sum(r['ref_n'] for r in rs)):8.2f} {silent:6d} {drums:6d} {first:7.2f}  {dict(sorted(shifts.items()))}")
    inst = defaultdict(int)
    for r in rows: inst[(r["group"], r["instrument"])] += 1
    print("\ninstrument labels:", {g: {i: n for (gg, i), n in sorted(inst.items()) if gg == g} for g in sorted(set(r["group"] for r in rows))})

if __name__ == "__main__":
    main()
