# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "safetensors", "huggingface_hub", "einops", "mido", "packaging", "soundfile"]
# ///
"""Teacher-forced logits from the upstream model (CPU fp32) for one chunk of a
reference directory written by Tools/reference_tokens.py --write-wav.

    uv run Tools/dump_logits.py <size> <reference clip dir> <chunk index>

Feeds the exact prefix, then the recorded prompt+tokens one at a time, and
records the logits at every step to <dir>/<size>/fixtures/logits_<chunk>.f32
(shape [len(prompt+tokens)+1, card]). LogitParityTests replays the same
sequence through the Core AI model and reports per-step PSNR, argmax flips and
the top-two margin at any flip. Needs a muscriptor checkout beside this repo
(or MUSCRIPTOR_DIR)."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
for c in ([Path(os.environ["MUSCRIPTOR_DIR"])] if os.environ.get("MUSCRIPTOR_DIR") else []) + [HERE.parent.parent / "muscriptor", HERE.parent / "muscriptor", Path.home() / "Developer" / "muscriptor"]:
    if c.exists(): sys.path.insert(0, str(c)); break
else: sys.exit("clone https://github.com/muscriptor/muscriptor beside this repo")
from huggingface_hub import hf_hub_download
from muscriptor.transcription_model import TranscriptionModel, _SAMPLE_RATE, _SEGMENT_DURATION
from muscriptor.modules.streaming import init_states, increment_steps

size, ref, ci = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
fx = ref / size / "fixtures"
c = json.load(open(fx / "chunks.json"))["chunks"][ci]
seq = c["prompt"] + c["tokens"]
tm = TranscriptionModel.load_model(weights_path=hf_hub_download(f"MuScriptor/muscriptor-{size}", "model.safetensors"), device="cpu", dtype="float32")
lm = tm._model
wav = tm._load_wav(str(ref / "audio_16k.wav"), None)
seg = int(_SEGMENT_DURATION * _SAMPLE_RATE)
chunk = wav[:, ci * seg:(ci + 1) * seg]
if chunk.shape[-1] < seg: chunk = torch.nn.functional.pad(chunk, (0, seg - chunk.shape[-1]))
cond = tm._build_conditions(chunk, None)
with torch.no_grad():
    prepared = lm.condition_provider.tokenize(cond); ct = lm.condition_provider(prepared)
    prepend = sum(v[0].shape[1] for v in ct.values())
    state = init_states(lm, batch_size=1, sequence_length=prepend + 2000)
    tokens = [lm.card] + seq
    recorded = []
    out = lm(torch.tensor([[tokens[0]]]), ct, first_step=True, model_state=state)   # prefix + initial token
    increment_steps(lm.transformer, state, increment=1 + prepend)
    recorded.append(out[0, -1].numpy().copy())
    for t in tokens[1:]:
        out = lm(torch.tensor([[t]]), ct, first_step=False, model_state=state)
        increment_steps(lm.transformer, state, increment=1)
        recorded.append(out[0, -1].numpy().copy())
logits = np.stack(recorded).astype(np.float32)   # [len(seq)+1, card]
logits.tofile(fx / f"logits_{ci}.f32")
# Sanity: greedy on these logits must reproduce the recorded stream (the prompt is forced).
start = len(c["prompt"])
agree = int((logits.argmax(1)[start:start + len(c["tokens"])] == np.array(c["tokens"])).sum())
print(f"{ref.name} {size} chunk {ci}: {logits.shape}, greedy reproduces {agree}/{len(c['tokens'])} generated tokens, prepend {prepend}")
