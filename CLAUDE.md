# CLAUDE.md — swift-music-transcriber

MuScriptor (Kyutai × Mirelo audio-to-MIDI) on Core AI. Module `MusicTranscriber`; the
CLI is `../swift-scribe-cli` (binary `scribe`). Read `README.md` for what it does and
`Tools/README.md` for how the model asset is made. Everything below is what a
re-derivation would get wrong.

## Build & test
```bash
swift build && swift test                     # 26 tests, golden fixtures, no model
SCRIBE_MODEL=~/Library/Application\ Support/scribe/models swift test   # + end-to-end parity
```
The end-to-end test needs a **medium fp32** asset and reproduces upstream's decode
token for token on `Fixtures/demo_16k.wav`. If it ever fails, `DiagnosticTests`
prints per-chunk first-divergence and the mel error.

## The Core AI traps (all measured 2026-09-23)
- **Default specialization crashes** in the Neural Engine MLIR pass on fp32 graphs
  (`mlir::anec`, "expected fp16"). Load with `preferredComputeUnitKind: .gpu`.
- **The runtime specializes per input shape** — ~180 ms for every new sequence
  length — unless `expectFrequentReshapes = true`. With it: one compile per asset,
  then ~1 ms/step on a toy. The specialized path was ALSO less accurate (26 dB vs
  68 dB in decode). Never turn it off.
- Cache views must come from `inout NDArray` parameters; the lifetime checker
  rejects views taken through a class property.
- `coreai.runtime` in Python segfaults on the stateful graph; verify in Swift.
- Swift bin path on this toolchain is `.build/out/Products/Release`.

## Upstream semantics that bit
- Prefix order is `[mel(501), dataset_null(1), instrument_group(L)]`: LMModel
  PREPENDS conditions in tokenize order, so the wav ends up first. The last mel
  frame is masked to zero (length mask covers 500 of 501).
- The 2,000-token budget counts the forced prelude, and the last budgeted token
  is never fed back. Feeding it back overflowed the 2503-slot cache on `small`.
- `AVAudioFile.length` was 611 frames short on a libsndfile float WAV; read to
  EOF (error −39) instead. It silently dropped the third chunk before.
- MIDI ticks: round on/off separately, half-to-even, from the tempo as whole
  microseconds — or one note in ten is a tick off from upstream.
- Program 96 is NOT drums (upstream's table aliases it by accident).

## Precision
fp32 is default: exact against upstream and no slower than fp16 for medium on
an M3 Max. fp16 diverges only where output is already degenerate (small on dense
material) — but on `large` it also moved the note count on the mp3 (161 vs 190),
so treat fp16 as a memory option, not an equal.

## Beat tracker
MusicUnderstanding, not beat_this. Same fitting maths. On the demo it reports
half the tempo beat_this does (77.5 vs 154.9 BPM). Documented, not "fixed".

## Fixtures
`Tools/dump_fixtures.py` regenerates `Tests/.../Fixtures` from a muscriptor
checkout; `Tools/reference_tokens.py` records upstream's per-chunk tokens for a
clip. Both need the checkout beside this repo and, for tokens, the HF login.
