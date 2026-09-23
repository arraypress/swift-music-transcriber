# CLAUDE.md — swift-music-transcriber

MuScriptor (Kyutai × Mirelo audio-to-MIDI) on Core AI. Module `MusicTranscriber`; the
CLI is `../swift-scribe-cli` (binary `scribe`). Read `README.md` for what it does and
`Tools/README.md` for how the model asset is made. Everything below is what a
re-derivation would get wrong.

## Build & test
```bash
swift build && swift test                     # 41 tests, golden fixtures, no model (2 parity tests skip without env)
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

## Beat tracker (0.4.0: Beat This! ported, upstream's)
`Tools/export_beat_this.py` → `scribe-beat-this-float32.aimodel` (80 MB).
`BeatTracker.automatic` uses it when installed, else MusicUnderstanding.
- The ORIGINAL module torch.exports and converts, but Core AI's compiler
  refuses it at load: `mps.strided_slice` infers `?x1x32x0` on the einops
  `b n h -> b h n 1` gate reshape with a dynamic time axis. Fix: one method
  re-authored (`Attention.forward`, plain reshape/permute, RoPE as a constant
  32×32 matrix, interleaved pairs like rotary_embedding_torch). Max |diff| 0.
- Front end: 22.05 kHz, n_fft 1024, hop 441, magnitude/√1024
  (torchaudio `normalized="frame_length"`), 128 Slaney mels 30–11000 Hz, no
  norm, log1p(1000·x). MuScriptor feeds it the 16 kHz signal; upstream
  resamples with soxr, we use the julius port — the one non-shared step.
  Beats still land within one frame on every test clip.
- Chunking is upstream's split_piece/aggregate (1500 frames, 6-frame borders,
  keep_first, last chunk shifted to the end). TEST TRAP: the fixture chunk is
  already padded; compare a single `run(chunk:)`, not `logits(mel:)`, or the
  double padding costs 10 dB.
- MEASURED on the 175 loops: Beat This! within 1 BPM 48/175, no grid 118/175;
  Apple 111 and 41. On the 2-min track Beat This! 178 (right), Apple 89.
  Short loops wobble past the 5% residual rule for Beat This!. `automatic`
  still prefers Beat This! (parity with upstream); loops want `--bpm name`.
- The tracker is cached per MusicTranscriber; loading per file cost ~1 s a loop.

## Tempo octave
`tempoRange:` / `--bpm-range lo-hi`: fleet multipliers (from swift-music-analysis)
snap the detection in; with a wide range two ratios can land (89 → 133.5 or
178) so the drum onsets' beat-level concentration picks. Measured: 178 on the
Makina track (user confirmed), 155 on the demo, 123 for the 82 bassline.

## Timing
The model's onsets DRIFT within a chunk: on the 123 BPM piano loop, 35 ms
early at 0.2 s converging to ~10 ms late by 3 s (frac-of-sixteenth 0.72 →
0.98 → 0.05). A constant lag correction (upstream's, ours) measures ~0 and
does nothing. `--quantize` on a fixed grid fixed all 43 onsets. Do not chase
"alignment" with an offset again.

## The STFT window is the checkpoint's, not a formula
Every checkpoint stores `...mel_spec_transform.spectrogram.window`: a periodic
Hann window rounded to half precision (two samples even sit one fp16 ulp off a
straight rounding). A float32 Hann window is 2.4e-4 away per sample, which
lifts the near-empty mel bins above 7 kHz by whole log units: the mel dropped
to 36 dB against upstream on real clips (99 dB with the stored window) and 8
of 19 parity runs flipped one near-tie token (margins 0.01–0.09 logits). The
window is tabulated in `Support/MelWindow.swift` and `FrontEndTests` checks it
bit for bit against `Fixtures/mel_window.f32`. The stored filterbank is a
float32 torchaudio bank (≤3.1e-4 from ours in double), below the STFT noise
floor; it is not embedded. Synthetic fixtures (sine + noise) never showed
this — only lowpassed real audio does.

## Fixtures and parity
`Tools/dump_fixtures.py` regenerates `Tests/.../Fixtures` from a muscriptor
checkout (mel pair through the checkpoint's own window and bank);
`Tools/reference_tokens.py --write-wav` records upstream's per-chunk tokens
and the exact 16 kHz samples for a clip; `Tools/dump_logits.py` records
upstream's teacher-forced logits for one chunk. All need the checkout beside
this repo (or `MUSCRIPTOR_DIR`) and, for the model, the HF login. Then:
```bash
SCRIBE_PARITY_DIR=<dir of clips> SCRIBE_MODEL=... swift test --filter ParityTests   # PASS/DIFF per clip×size, note F1
SCRIBE_LOGIT_CASES="medium:<dir>/<clip>:0;…" SCRIBE_MODEL=... swift test --filter LogitParityTests   # per-step PSNR, flips, margins
```
