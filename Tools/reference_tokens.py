# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "safetensors", "huggingface_hub", "einops", "mido", "packaging", "soundfile"]
# ///
"""The upstream reference for a clip: per-chunk prompts and greedy tokens from
MuScriptor's own pipeline (CPU, fp32, prelude forcing), plus the log-mel of
each chunk and the MIDI it writes.

    uv run Tools/reference_tokens.py --size medium --audio clip.wav --out ref/

Feed the resulting fixtures/chunks.json to Tools/dump_fixtures.py, or compare
against `MusicTranscriber.tokenObserver` output. Needs a muscriptor checkout
beside this repo and a Hugging Face login with the licence accepted."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
import os
for c in ([Path(os.environ["MUSCRIPTOR_DIR"])] if os.environ.get("MUSCRIPTOR_DIR") else []) + [HERE.parent.parent / "muscriptor", HERE.parent / "muscriptor", Path.home() / "Developer" / "muscriptor"]:
    if c.exists(): sys.path.insert(0, str(c)); break
else: sys.exit("clone https://github.com/muscriptor/muscriptor beside this repo")
from huggingface_hub import hf_hub_download
from muscriptor.transcription_model import TranscriptionModel, _SAMPLE_RATE, _SEGMENT_DURATION
from muscriptor.events import ChunkBoundary, ProgressEvent, decode_model_tokens
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="small")
ap.add_argument("--audio", required=True)
ap.add_argument("--out", default=str(HERE / "reference"))
ap.add_argument("--max-chunks", type=int, default=None)
ap.add_argument("--reuse", action="store_true", help="reuse chunks.json/mel_*.f32 already in the fixtures dir instead of re-running the upstream decode")
ap.add_argument("--instruments", help="comma-separated upstream group names: a conditioned + masked reference")
ap.add_argument("--write-wav", action="store_true", help="also write <out>/audio_16k.wav, the exact samples upstream decoded, for the Swift side to read")
args = ap.parse_args()

weights = hf_hub_download(f"MuScriptor/muscriptor-{args.size}", "model.safetensors")
hf_hub_download(f"MuScriptor/muscriptor-{args.size}", "config.json")
tm = TranscriptionModel.load_model(weights_path=weights, device="cpu", dtype="float32")
lm, tok = tm._model, tm._tokenizer
dims = dict(dim=lm.dim, heads=lm.transformer.layers[0].self_attn.num_heads, layers=len(lm.transformer.layers),
            card=lm.card, max_ctx=503 + 2000)
print("[real] loaded", args.size, dims, flush=True)

# --- audio -> chunks -> conditions (exactly as transcribe() does)
wav = tm._load_wav(args.audio, None)
seg = int(_SEGMENT_DURATION * _SAMPLE_RATE)
n_chunks = int(np.ceil(wav.shape[-1] / seg))
if args.max_chunks: n_chunks = min(n_chunks, args.max_chunks)
from muscriptor.tokenizer.mt3 import instrument_group_from_names
import soundfile as sf
names = [n for n in (args.instruments or "").split(",") if n.strip()]
instrument_group = instrument_group_from_names(names) if names else None
forbidden = torch.tensor(tok.forbidden_token_ids(names), dtype=torch.long) if names else None
out_root = Path(args.out)
if args.write_wav:
    out_root.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_root / "audio_16k.wav"), wav[0].numpy(), _SAMPLE_RATE, subtype="FLOAT")
conds, seek_times, mels = [], [], []
mel_cond = lm.condition_provider.conditioners["self_wav"]
for i in range(n_chunks):
    chunk = wav[:, i * seg:(i + 1) * seg]
    if chunk.shape[-1] < seg: chunk = F.pad(chunk, (0, seg - chunk.shape[-1]))
    c = tm._build_conditions(chunk, instrument_group)[0]
    conds.append(c); seek_times.append(i * _SEGMENT_DURATION)
    with torch.no_grad():
        mels.append(mel_cond._mel_embedding(mel_cond.tokenize(c.wav["self_wav"])))   # [1,501,512] log-mel
print(f"[real] {wav.shape[-1] / _SAMPLE_RATE:.1f}s audio -> {n_chunks} chunks", flush=True)

out = Path(args.out) / args.size
fixtures = out / "fixtures"; fixtures.mkdir(parents=True, exist_ok=True)
if args.reuse and (fixtures / "chunks.json").exists():
    chunks = json.load(open(fixtures / "chunks.json"))["chunks"]
    print(f"[real] reusing {len(chunks)} reference chunks from {fixtures}", flush=True)
else:
    # --- reference token stream from the upstream generator, recording each chunk's forced prompt
    prompts, generated = [], []
    _orig_generate = lm.generate
    def recording_generate(*a, **kw):
        p = kw.get("prompt")
        prompts.append([] if p is None else p[0].tolist())
        yield from _orig_generate(*a, **kw)
    lm.generate = recording_generate
    t0 = time.time()
    stream = list(tm._generate_token_stream(conds, seek_times, 1, 2000, False, 1.0, 1.0, True, True, 1, forbidden))
    lm.generate = _orig_generate
    print(f"[real] upstream greedy decode on CPU: {time.time() - t0:.1f}s", flush=True)
    cur = None
    for item in stream:
        if isinstance(item, ChunkBoundary):
            cur = []; generated.append(cur)
        elif isinstance(item, ProgressEvent):
            continue
        else:
            cur.append(int(item))
    # generated[i] starts with the forced prompt tokens (they flow through the stream); strip them
    chunks = []
    for i, (p, g) in enumerate(zip(prompts, generated)):
        assert g[:len(p)] == p, f"chunk {i}: prompt not at head of stream"
        chunks.append({"prompt": p, "tokens": g[len(p):]})
        print(f"[real] chunk {i}: prompt {len(p)} tokens, generated {len(g) - len(p)} tokens", flush=True)

    # reference MIDI, for listening later
    events = list(decode_model_tokens(iter(stream), tok._vocab, tm._instrument_for_program, frame_rate=tok.frame_rate))
    (fixtures / "ref.mid").write_bytes(tm.events_to_midi_bytes(iter(events)))
    notes = sum(1 for e in events if type(e).__name__ == "NoteStartEvent")
    print(f"[real] reference: {notes} notes -> {fixtures / 'ref.mid'}", flush=True)


for i, m in enumerate(mels):
    m.numpy().astype(np.float32).tofile(fixtures / f"mel_{i}.f32")
json.dump({"dims": dims, "eos": tok.eos_id, "initial": lm.card, "inst_tokens": [0], "max_gen_len": 2000,
           "instruments": names, "size": args.size,
           "chunks": chunks, "audio": str(Path(args.audio).name)}, open(fixtures / "chunks.json", "w"))
print("[real] fixtures ->", fixtures, flush=True)
