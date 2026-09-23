# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "safetensors", "huggingface_hub", "einops", "mido", "packaging", "soundfile"]
# ///
"""Golden fixtures for the Swift tests, straight from the upstream code.

    uv run Tools/dump_fixtures.py <audio> Tests/MusicTranscriberTests/Fixtures [chunks.json]

Writes the mel front-end pair, the resampler pairs, and — given a chunks.json
from Tools/reference_tokens.py — the decoder event stream, the cleaned notes
and the MIDI upstream writes for them. The repo's fixtures came from the 10 s
demo clip in the muscriptor checkout (web/public/headache_by_lost_deposit_10s.mp3)
and the medium fp32 CPU run. Needs a muscriptor checkout beside this repo.

The mel pair uses the STFT window and filterbank buffers stored in the
checkpoint (identical across the three sizes), not freshly computed ones: the
stored window is a half-precision rounding of torch.hann_window(2048), and a
float32 Hann window moves the near-empty top mel bins by whole log units."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
import os
for c in ([Path(os.environ["MUSCRIPTOR_DIR"])] if os.environ.get("MUSCRIPTOR_DIR") else []) + [HERE.parent.parent / "muscriptor", HERE.parent / "muscriptor", Path.home() / "Developer" / "muscriptor"]:
    if c.exists(): sys.path.insert(0, str(c)); break
else: sys.exit("clone https://github.com/muscriptor/muscriptor beside this repo")
from muscriptor.utils.audio import load_audio
from muscriptor.utils.resample import resample_frac
from muscriptor.modules.mel_spectrogram import melscale_fbanks
from muscriptor.modules.conditioners import MelSpectrogramConditioner, WavCondition
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import DRUM_PROGRAM
from muscriptor.events import ChunkBoundary, decode_model_tokens, NoteStartEvent, NoteEndEvent
from muscriptor.transcription_model import _build_instrument_for_program

audio, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

# 1. mel: the first 5 s chunk of the clip at 16 kHz, and its log-mel, through
#    the window and filterbank the checkpoint actually carries
from huggingface_hub import hf_hub_download
from safetensors import safe_open
with safe_open(hf_hub_download("MuScriptor/muscriptor-medium", "model.safetensors"), "pt") as f:
    prefix = "condition_provider.conditioners.self_wav.mel_spec_transform."
    stored_window = f.get_tensor(prefix + "spectrogram.window").float()
    stored_fb = f.get_tensor(prefix + "mel_scale.fb").float()
stored_window.numpy().astype(np.float32).tofile(out / "mel_window.f32")
wav = load_audio(audio, target_sr=16000)            # [1, T]
chunk = wav[:, :80000]
chunk.numpy().astype(np.float32).tofile(out / "mel_input.f32")
cond = MelSpectrogramConditioner(output_dim=8, device="cpu", sample_rate=16000, n_fft=2048, frame_rate=100,
                                 n_mel_bins=512, log_scale=True, eps=1e-6, normalize_audio=False)
cond.mel_spec_transform.spectrogram.window.data = stored_window
cond.mel_spec_transform.mel_scale.fb.data = stored_fb
mel = cond._mel_embedding(WavCondition(chunk.unsqueeze(0), torch.tensor([80000]), [16000], [None], [0.0]))
mel.numpy().astype(np.float32).tofile(out / "mel_output.f32")
stored_fb.numpy().astype(np.float32).tofile(out / "mel_filterbank.f32")
assert (melscale_fbanks(1025, 0.0, 8000.0, 512, 16000) - stored_fb).abs().max() < 1e-3   # the bank is a fresh torchaudio bank up to float32 noise
print("mel:", tuple(mel.shape))

# 2. resampler: half a second of noise-plus-tone at 44.1 kHz and 48 kHz -> 16 kHz
for sr in (44100, 48000):
    t = torch.arange(int(0.5 * sr)) / sr
    x = (0.5 * torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn(len(t))).float()
    y = resample_frac(x, sr, 16000)
    x.numpy().tofile(out / f"resample_{sr}_in.f32"); y.numpy().tofile(out / f"resample_{sr}_out.f32")
    print(f"resample {sr}: {len(x)} -> {len(y)}")

# 3. decoder: upstream's event stream for a recorded token stream
if len(sys.argv) > 3:
    chunks = json.load(open(sys.argv[3]))["chunks"]
    tok = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
    lookup = _build_instrument_for_program(tok)
    def stream():
        for i, c in enumerate(chunks):
            yield ChunkBoundary(i * 5.0, (i + 1) * 5.0 if i + 1 < len(chunks) else None)
            yield from c["prompt"]; yield from c["tokens"]
    events = []
    for e in decode_model_tokens(stream(), tok._vocab, lookup, frame_rate=100):
        if isinstance(e, NoteStartEvent):
            events.append({"type": "start", "pitch": e.pitch, "start_time": e.start_time, "index": e.index, "instrument": e.instrument})
        elif isinstance(e, NoteEndEvent):
            events.append({"type": "end", "end_time": e.end_time, "start_event_index": e.start_event_index})
    json.dump({"chunks": chunks, "events": events}, open(out / "decoder.json", "w"))
    print("decoder events:", len(events))
    # 4. notes after upstream's cleanup, and the MIDI it writes (no tempo grid)
    from muscriptor.tokenizer.notes import Note, validate_notes, trim_overlapping_notes
    from muscriptor.utils.midi import notes_to_midi
    notes, opened, names = [], {}, {}
    for e in events:
        if e["type"] == "start":
            is_drum = e["instrument"] == "drums"
            program = DRUM_PROGRAM if is_drum else next(p for p, n in [(g[0], name) for name, g in [(n, tok.group_program_map[i]) for n, i in __import__("muscriptor.tokenizer.mt3", fromlist=["x"]).MT3_FULL_PLUS_GROUP_NAMES.items() if i in tok.group_program_map]] if n == e["instrument"]) if not e["instrument"].startswith("program_") else int(e["instrument"][8:])
            names[program] = e["instrument"].replace("_", " ")
            opened[e["index"]] = Note(is_drum=is_drum, program=program, onset=e["start_time"], offset=e["start_time"], pitch=e["pitch"])
        else:
            n = opened.pop(e["start_event_index"]); n.offset = e["end_time"]; notes.append(n)
    notes = trim_overlapping_notes(validate_notes(notes, fix=True), sort=True)
    json.dump([{"pitch": n.pitch, "onset": n.onset, "offset": n.offset, "program": n.program, "is_drum": n.is_drum} for n in notes],
              open(out / "notes.json", "w"))
    midi = notes_to_midi(notes, program_names=names)
    midi.save(str(out / "reference.mid"))
    print("notes:", len(notes), "-> reference.mid")
