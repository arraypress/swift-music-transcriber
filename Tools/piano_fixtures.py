# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "piano_transcription_inference", "torchlibrosa", "librosa", "matplotlib", "mido", "audioread", "soundfile"]
# ///
"""Upstream reference fixtures for the piano engine.

    uv run Tools/piano_fixtures.py --checkpoint Tools/exports/piano_checkpoint.pth name=clip.wav [name2=clip2.wav] [--out dir]

Per clip: `audio_16k.wav` (the exact samples upstream decoded), every intermediate tensor of the
first segment (`logmel_0`, `<branch>.trunk_0`, `<branch>.gru_0`), the seven stitched framewise
outputs, `events.json` (notes, pedals, audio length) and `upstream.mid`. `PianoTests` reads the
piano-loop set from `Tests/.../Fixtures/Piano` (renamed without dots: `frame_trunk_0.f32`,
`frame_gru_0.f32`)."""
import argparse, json, os, sys, time
import numpy as np, torch, soundfile as sf, librosa
from piano_transcription_inference import PianoTranscription, sample_rate

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", default="Tools/reference/piano")
ap.add_argument("clips", nargs="+", help="name=path.wav")
args = ap.parse_args()
tr = PianoTranscription(device="cpu", checkpoint_path=args.checkpoint); model = tr.model
for spec in args.clips:
    name, path = spec.split("=", 1)
    y, sr = sf.read(path, dtype="float32"); y = y.mean(1) if y.ndim > 1 else y
    if sr != sample_rate: y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate, res_type="soxr_hq").astype(np.float32)
    d = os.path.join(args.out, name); os.makedirs(d, exist_ok=True); sf.write(f"{d}/audio_16k.wav", y, sample_rate, subtype="FLOAT")
    caps, hooks = {}, []
    def keep(key):
        def h(m, i, o): caps.setdefault(key, []).append((o if torch.is_tensor(o) else o[0]).detach().clone())
        return h
    def keep_in(key):
        def h(m, i): caps.setdefault(key, []).append(i[0].detach().clone())
        return h
    hooks.append(model.note_model.logmel_extractor.register_forward_hook(keep("logmel")))
    for owner, names in [(model.note_model, ["frame_model", "reg_onset_model", "reg_offset_model", "velocity_model"]),
                         (model.pedal_model, ["reg_pedal_onset_model", "reg_pedal_offset_model", "reg_pedal_frame_model"])]:
        for bname in names:
            b = getattr(owner, bname)
            hooks.append(b.gru.register_forward_pre_hook(keep_in(f"{bname}.trunk"))); hooks.append(b.gru.register_forward_hook(keep(f"{bname}.gru")))
    t = time.time(); out = tr.transcribe(y, f"{d}/upstream.mid"); took = time.time() - t
    for h in hooks: h.remove()
    od = out["output_dict"]
    print(f"{name}: {len(y)/sample_rate:.1f} s, {took:.1f} s CPU; segments {len(caps['logmel'])}; notes {len(out['est_note_events'])} pedals {len(out['est_pedal_events'])}")
    for k, v in caps.items():
        v[0].numpy().astype(np.float32).tofile(f"{d}/{k}_0.f32")
    for k, v in od.items():
        if hasattr(v, "shape"): np.asarray(v, dtype=np.float32).tofile(f"{d}/{k}.f32")
    json.dump({"notes": out["est_note_events"], "pedals": out["est_pedal_events"], "audio_len": len(y),
               "shapes": {k: list(v.shape) for k, v in od.items() if hasattr(v, "shape")}},
              open(f"{d}/events.json", "w"), indent=1, default=float)
