# Tools

Python, run with `uv run` (each script declares its own dependencies). All three need a
checkout of [muscriptor](https://github.com/muscriptor/muscriptor) beside this repo (or
`--muscriptor`); `export.py` also needs [apple/coreai-models](https://github.com/apple/coreai-models).

| script | what |
|---|---|
| `export.py` | MuScriptor checkpoint → `scribe-<size>-<precision>.aimodel`. `--install` copies it into `~/Library/Application Support/scribe/models`. |
| `export_beat_this.py` | Beat This! `final0` (the tracker MuScriptor uses; MIT) → `scribe-beat-this-float32.aimodel`. One method re-authored for Core AI's compiler; asserts it matches the original before exporting. `--install` copies it beside the transcriber models. |
| `export_piano.py` | ByteDance piano transcription (Kong et al. 2020, Apache 2.0) → `scribe-piano-float32.aimodel`: the conv trunks in the graph, the GRU and head weights as a flat `parameters` entry point (Core AI has no recurrent op; the recurrences run in Swift). Downloads the checkpoint from Zenodo. `--install`. |
| `piano_fixtures.py` | Records upstream's intermediate tensors, stitched outputs, events and MIDI for clips, for `PianoTests`. |
| `dump_fixtures.py` | Regenerates the golden test fixtures (mel, resampler, decoder events, notes, MIDI) from the upstream code. |
| `reference_tokens.py` | Records upstream's per-chunk prompts and greedy tokens for a clip (`--write-wav` also keeps the exact 16 kHz samples), for `ParityTests` against `MusicTranscriber.tokenObserver`. |
| `score_midi.py` | Scores `--format json` output against MIDI ground truth (mir_eval onset+pitch F1, exact / octave / any-shift / chroma / onset-only, first-note recall) from a pairs manifest. |
| `dump_logits.py` | Records upstream's teacher-forced logits for one chunk of such a clip, for `LogitParityTests` (per-step PSNR, argmax flips, top-two margins). |

## How the export works

MuScriptor's decoder is re-authored in `export.py` with the same weights and maths but the
attention cache as explicit Core AI state (Apple's `coreai_models.primitives.macos.cache`), and
exported through `torch.export` → `TorchConverter` as **one asset with three functions**:

```
main   (inputs_embeds [1,Q,D], position_ids [1,S]; state k_cache, v_cache [L,1,H,2503,Dh]) -> logits [1,1,card]
prefix (mel [1,F,512] log-mel, inst_tokens [1,L] int32)                                    -> embeds [1,F+1+L,D]
embed  (tokens [1,T] int32)                                                                -> embeds [1,T,D]
```

The Swift side computes the log-mel (vDSP), calls `prefix` once per chunk, `embed` for the
prompt and for each generated token, and `main` with the cache state threaded through.

Export takes 10–30 s; the first Swift load compiles for the GPU (a few seconds, cached by the
runtime afterwards). Verified 2026-09-23: medium fp32 and large fp16 assets reproduce upstream's
greedy CPU decode token for token on the demo clip; small fp16 diverges where upstream's own
fp16 run does.

```sh
uv run Tools/export.py --size medium --install
uv run Tools/export.py --size large --dtype float16
uv run Tools/export_beat_this.py --install
uv run Tools/dump_fixtures.py ../muscriptor/web/public/headache_by_lost_deposit_10s.mp3 Tests/MusicTranscriberTests/Fixtures reference/medium/fixtures/chunks.json
uv run Tools/reference_tokens.py --size medium --audio <clip.wav> --out reference/<clip> --write-wav   # then dump_logits.py medium reference/<clip> 0
```

For an exact-truth check, render MIDI with the library's own synthesiser and score the
transcription of the render: `Auralizer.synthesize(midi: Data(contentsOf: url))` gives 44.1 kHz
samples, `AudioWriter.writeWAV` writes them — ten lines of Swift in a scratch package — then
`scribe <dir> --format json --detect-tempo off --out <run>` and `score_midi.py <run> --manifest`.
749 renders scored 0.947 on 2026-09-23.

`dump_fixtures.py` takes the STFT window and mel filterbank from the checkpoint itself (they are
identical in all three sizes); the stored window is a half-precision rounding of
`torch.hann_window(2048)`, and the Swift front end carries that exact buffer (`MelWindow.swift`).

The toolchain is beta (`coreai-torch 0.4.2`, `coreai-core 1.0.0b2`, torch 2.13, Python 3.11/3.12).
The Python runtime in `coreai-core` segfaults on the stateful graph; the Swift tests are the gate.
