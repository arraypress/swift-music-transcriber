# Tools

Python, run with `uv run` (each script declares its own dependencies). All three need a
checkout of [muscriptor](https://github.com/muscriptor/muscriptor) beside this repo (or
`--muscriptor`); `export.py` also needs [apple/coreai-models](https://github.com/apple/coreai-models).

| script | what |
|---|---|
| `export.py` | MuScriptor checkpoint → `scribe-<size>-<precision>.aimodel`. `--install` copies it into `~/Library/Application Support/scribe/models`. |
| `dump_fixtures.py` | Regenerates the golden test fixtures (mel, resampler, decoder events, notes, MIDI) from the upstream code. |
| `reference_tokens.py` | Records upstream's per-chunk prompts and greedy tokens for a clip, for parity work against `MusicTranscriber.tokenObserver`. |

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
uv run Tools/dump_fixtures.py ../muscriptor/web/public/headache_by_lost_deposit_10s.mp3 Tests/MusicTranscriberTests/Fixtures reference/medium/fixtures/chunks.json
```

The toolchain is beta (`coreai-torch 0.4.2`, `coreai-core 1.0.0b2`, torch 2.13, Python 3.11/3.12).
The Python runtime in `coreai-core` segfaults on the stateful graph; the Swift tests are the gate.
