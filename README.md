# swift-music-transcriber

Multi-instrument music transcription for Swift: a recording in, a MIDI file with one track per
instrument out, entirely on this machine. It is [MuScriptor](https://github.com/muscriptor/muscriptor)
— Kyutai and Mirelo's open transcription model — re-authored for Apple's Core AI runtime.

```swift
import MusicTranscriber

let transcriber = try await MusicTranscriber(model: try ModelLocator.resolve())   // medium, fp32
let result = try await transcriber.transcribe(url)                                 // any audio AVFoundation reads

result.notes.count                       // 173
result.instruments                       // ["distorted_electric_guitar", "electric_bass", "drums"]
result.beatGrid?.bpm                     // 77.5, or nil when no steady beat was found

// A loop whose name carries its tempo: skip the tracker, write 123 BPM in 4/4, never shift it.
let bpm = FilenameTempo.bpm(inFilename: url.lastPathComponent)          // 123 for "VENDOR_PK2_123_bass_loop_C.wav"
let loop = try await transcriber.transcribe(url, fixedTempo: bpm)

let midi = try MIDIAssembly.data(notes: result.notes, grid: result.beatGrid)
try midi.write(to: url.deletingPathExtension().appendingPathExtension("mid"))
```

Or as events, as they are decoded:

```swift
for try await event in try transcriber.events(url) {
    if case .noteStart(let note) = event { print(note.startTime, note.instrument, note.pitch) }
}
```

## What the model does

A mix goes in as 5-second chunks of 16 kHz mono. For each chunk the model writes a stream of
note events — pitch, on, off and one of 36 instrument groups — at 10 ms resolution, and each chunk
opens by declaring which notes are still sounding from the one before, so a sustained chord
survives the boundary. There is **no velocity, pedal or pitch bend**: the model's vocabulary has
one bit for on-versus-off, so every note is written at velocity 100. The type says so
(`TranscribedNote` has no velocity field) rather than inventing a number.

Three published checkpoints, all the same architecture:

| | layers / width | asset (fp32) | on the 10 s demo, M3 Max |
|---|---|---|---|
| `small` | 14 / 768 | 412 MB | 4.5 ms per token; hits the token budget on dense material |
| `medium` (default) | 24 / 1024 | 1.2 GB | 10 ms per token, **2.3× realtime**, 173 notes |
| `large` | 48 / 1536 | 5.5 GB | fp16 **1.1× realtime**, fp32 0.5× — the one size where fp16 pays |

## Measured against upstream

Every stage was held to the Python code, not to a description of it:

| stage | test | result |
|---|---|---|
| log-mel front end (vDSP) | upstream's mel through the checkpoint's own window and filterbank | 99 dB PSNR on real clips |
| sinc resampler (julius port) | fixtures at 44.1 and 48 kHz | > 90 dB PSNR |
| decoder logits, teacher-forced | upstream's CPU fp32 logits at every step of 4 chunks | ≥ 116 dB with upstream's mel, ≥ 94 dB with ours; no argmax flips |
| token decoder, tie prologue, note cleanup | upstream's event stream for 3 real chunks | identical, 346 events |
| MIDI layout | upstream's file for the same notes | same notes, ticks, tracks, tempo |
| **whole pipeline**, all three sizes, fp32 | upstream's greedy CPU decode on 7 clips (bass, piano, drums, disco, a 2-minute makina track) | **token for token in all 19 runs, 25,000+ tokens** |
| large fp16 | same, demo clip | token for token |
| small fp16 | same | diverges after ~50 tokens — where upstream's own fp16 run diverges |

One thing had to be taken from the checkpoint rather than recomputed: the STFT window. Every
checkpoint stores a periodic Hann window rounded to half precision, and a float32 Hann window
lifts the near-empty mel bins above 7 kHz by whole log units. Before the stored window was used,
8 of the 19 runs flipped a single near-tie token (top-two margin 0.01–0.09 logits). The window is
tabulated in `MelWindow.swift` and checked bit for bit against the checkpoint.

So fp32 is the default. On this hardware it was no slower than fp16 for `medium`; for `large`,
which is bandwidth-bound at 5.5 GB, fp16 is twice as fast and still exact on the demo.

**The MP3 decoder is a bigger variable than the precision.** The parity rows above feed the
model the same 16 kHz samples upstream decoded with libsndfile. Given the MP3 itself,
AVFoundation decodes it slightly differently, and on `large` that moved the transcription from
190 notes to 161 (medium was unchanged). For reproducible results feed WAV or FLAC.

**Tempo can be an octave out, and the notes can fix it.** The tracker hears the half-time pulse
on fast electronic music — 89 for a 178 BPM Makina track, 77.5 for the 155 BPM demo — and
two-thirds on some basslines. `tempoRange:` declares where the tempo plausibly lives and the
detection is snapped in by the smallest musical ratio; when two ratios land (89 reaches
120…190 as 133.5 and as 178) the transcribed drum hits decide, because at the true tempo they
sit on the beats. That is a measurement on the model's own output, not a guess: 178 and 155
on those two, and a 123 BPM bassline heard at 82 comes back as 123.

**The beat tracker is upstream's too, when installed.** MuScriptor finds tempo and metre with
[Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, MIT), and `Tools/export_beat_this.py`
puts that model's `final0` checkpoint on Core AI as well: 20M parameters, 80 MB, a few seconds
per song. Its spectrogram front end, chunking and peak-picking are ported and held to fixtures
from the Python code, and on the demo, the Makina track and two loops it returns upstream's
beats to the frame — including a loop where upstream's fit fails and falls back to 120 BPM, which
this does too, because matching upstream includes matching its misses. Without the asset, Apple's
MusicUnderstanding (via [swift-music-analysis](https://github.com/arraypress/swift-music-analysis))
stands in; on the demo it hears the half-time pulse, 77.5 where Beat This! says 154.9.
`BeatTracker` picks: `.automatic` (Beat This! when installed), `.beatThis`, `.apple`.

| tracker | demo | Makina track (178) | bass loop (123) | piano loop (123) |
|---|---|---|---|---|
| Beat This! (upstream's, ported) | 154.9 = upstream | 178.0 = upstream | 122.9 = upstream | no fit = upstream |
| Apple MusicUnderstanding | 77.5 | 89.0 | 81.9 | 123.1 |

## Measured on 175 loops

A sample of the user's loops from a commercial pack (7.6 s, 44.1 kHz, BPM and root in the
filename), medium fp32, defaults, one `scribe` run per folder. Tempo is MusicUnderstanding's;
key root is the fleet's `midi key` run over the transcription against the filename.

| folder | files | notes / file | tempo within 1 BPM | no grid | key root | what the model called it |
|---|---|---|---|---|---|---|
| bass loops | 40 | 32 | 31 | 0 | 26 / 38 | electric_bass 32, piano 4, guitar 5 |
| full drum loops | 8 | 21 | 8 | 0 | — | drums 7 |
| stripped drum loops | 8 | 23 | 8 | 0 | — | drums 6 |
| hat loops | 8 | 18 | 5 | 3 | — | drums 6 |
| bongo / shaker loops | 16 | 1–4 | 4 | 12 | — | mostly nothing |
| synth loops | 20 | 37 | 16 | 1 | 8 / 16 | acoustic_guitar 7, distorted guitar 4, piano 3 |
| filtered disco loops | 20 | 90 | 13 | 3 | — | drums 9, electric_bass 8, guitar 7 |
| fx loops | 15 | 62 | 10 | 2 | 4 / 9 | distorted guitar 8, drums 6 |
| vocal loops | 40 | 18 | 16 | 20 | — | guitar 4, voice 3, drums 3 |
| **all** | **175** | | **111** | **41** | **38 / 63** | **4.5× realtime**, load and beat tracking included |

Three things the table says plainly:

- **Two synths in one class share one track, and only a better model hears them apart.** On a
  Makina track, `medium` put the lead and the pad on one piano track; `large`, unforced, wrote
  a separate `synth_lead` track of 338 notes. Forcing a synth-only mask on `medium` split them
  too, but shoved 931 melodic notes onto the bass — that is coercion, not accuracy. For full
  mixes where parts merge, use `large`.
- **Synths are not a class the model reaches for.** Twenty synth loops, zero `synth_lead` or
  `synth_pad` labels; they come back as guitars and pianos. With
  `--instruments synth_lead,synth_pad,electric_bass,drums` the same loops produce **48 notes
  per file instead of 37**, labelled by construction. On electronic material, mask.
- **Untuned percussion barely registers.** Bongo and shaker loops yield one to four notes; kit
  loops yield twenty and are labelled drums. Vocals are transcribed as pitch (18 notes per loop)
  but rarely labelled `voice` (3 of 40).
- **On eight-second loops, Apple's tracker beats upstream's.** Rerun with Beat This!: within
  1 BPM on 48 loops instead of 111, and no steady grid on 118 instead of 41 — its beats wobble
  past the 5% residual rule on short material that MusicUnderstanding fits. On the two-minute
  track the order reverses (178 versus 89). Songs: Beat This!. Loops: `--bpm name`.
- **Tempo holds where there is a beat.** With Apple's tracker, within 1 BPM on every kit loop and 31 of 40 bass loops;
  the 41 loops with no grid are mostly shakers, bongos and half the vocals, where a tracker has
  nothing to hold on to. Of the 23 wrong tempos, 8 were exactly 2/3 (a triplet bassline tracked
  on its subdivision) and 5 were half. Loops carry their tempo in the name, so `fixedTempo:`
  with `FilenameTempo` sidesteps the tracker entirely — and makes `quantize` exact.
- **The model's timing is not a constant offset.** On a 123 BPM piano loop its onsets run up to
  35 ms early for the first three seconds of a chunk and settle about 10 ms late after that.
  Upstream's lag correction, which fits one shift, measures the two halves as cancelling and
  applies nothing; nothing else could either. `quantize` against the loop's own grid put every
  one of its 43 onsets exactly on a sixteenth. For loops, quantize.

## Measured against MIDI ground truth

712 loops from four commercial sample packs that ship the MIDI they were played from: bass
loops, acid lines, pads, synth lines and stabs, plus a dozen construction-kit stems. Medium fp32,
defaults, raw events (`--format json`, no tracker), scored with `mir_eval` the standard way —
a note is correct when its onset is within 50 ms and its pitch within 50 cents; offsets are
ignored. `Tools/score_midi.py` is the scorer.

Two things about the ground truth first, because they set the ceiling:

- **The MIDI is what was played, not what sounds.** A bass patch plays one or two octaves
  below the written note on 340 of 712 files, so the honest pitch score allows a per-file
  whole-octave shift (the "octave" column; "exact" is there to show the gap). Chord-stab,
  gated and octave-layered patches make more notes sound than the MIDI holds, which the model
  transcribes and the score counts against it: the model writes 1.8 notes for every MIDI note.
- **The model does not hear everything.** 59 files produced nothing: sub-only basslines (70–93%
  of their energy below 120 Hz) and quiet, bright, percussive synth textures. Peak-normalising
  three of the latter changed nothing, so it is not a level problem.

| loops | files | octave F1 | precision | recall | onset-only F1 | first note found |
|---|---|---|---|---|---|---|
| bass (pack A) | 90 | 0.69 | 0.64 | 0.81 | 0.74 | 73% |
| bass (pack B) | 90 | 0.63 | 0.53 | 0.87 | 0.68 | 71% |
| synth lines (pack B) | 90 | 0.57 | 0.49 | 0.74 | 0.62 | 67% |
| synth loops, several parts each (pack C) | 160 | 0.57 | 0.50 | 0.77 | 0.62 | 71% |
| acid lines | 90 | 0.49 | 0.44 | 0.63 | 0.57 | 64% |
| synth loops (pack A) | 90 | 0.42 | 0.38 | 0.57 | 0.46 | 47% |
| pads | 90 | 0.33 | 0.28 | 0.44 | 0.38 | 72% |
| construction-kit stems | 12 | 0.31 | 0.30 | 0.33 | 0.35 | 0% |
| **all** | **712** | **0.53** | **0.47** | **0.69** | **0.58** | **69%** |

Matched onsets sit a median 6 ms late. Recall is the number to read: on monophonic bass the
model finds 81–87% of the written notes, and most of what it adds is the patch's own sub-octave
layer. Pads are the weak class — sustained chords come back as repeated notes, three for one.

**The same MIDI rendered with a General MIDI piano scores 0.95.** To separate the model from the
patches, every MIDI file was rendered through the library's own `Auralizer` (Apple's GM
synthesiser, the file's program, no effects) and transcribed: 749 renders, exact pitch, **note
F1 0.947, precision 0.96, recall 0.94**, 0.97 notes written per note played, first note found
91%, 14 silent files (very sparse or very high parts). Pads 0.93, everything else 0.93–0.98. So
when the audio contains exactly the written notes the model finds them; the gap to 0.53 on the
packs is what the patches add and what the timbres hide.

**Large versus medium**, same 120 real loops: 0.50 → 0.52 overall, but on the cleanest bass
pack 0.63 → **0.78** with precision 0.60 → 0.79 — large writes 1.06 notes per MIDI note where
medium writes 1.32, because it stops transcribing the sub-octave layer as a second note. Pads
and the harder synth pack lose a little. Large costs 4× the time; on bass it pays.

**The first note of a loop was a decoding rule, not a hearing problem.** The model reports a
note that is already sounding at a chunk's first frame in the chunk's tie prologue, and a loop
that starts on beat one begins that way. Upstream's event builder ignores a tie for a note that
is not open — which at the start of a recording is every one of them — so with upstream's
reading the first note was found on 3% of 235 loops. `leadingTies` (`--leading-ties`) reads it
as a note starting at time zero; the tokens are identical. Same 235 loops: first note 64%,
octave F1 0.54 → 0.56. `drop` reproduces upstream's stream exactly.

## The piano engine

For solo piano there is a second model: **High-resolution Piano Transcription with Pedals**
(Kong, Li, Song, Hou, Wang — ByteDance, 2020; Apache 2.0; 43M parameters), the one that hears
what MuScriptor cannot: a velocity for every note and the sustain pedal. `PianoTranscriber`
loads `scribe-piano-float32.aimodel` (136 MB) and returns notes with velocity plus pedal events;
`MIDIAssembly` writes them as note velocities and controller 64 on one acoustic-piano track,
with the same tempo-grid handling as the MuScriptor path. In the CLI: `--engine piano`.

Core AI has no recurrent op, and this model is half GRU: a decomposed GRU unrolls to about
770,000 ops at its size. So the asset carries the convolutional trunks as a graph and every
recurrent and head weight as a flat vector, and the sixteen bidirectional GRUs run in Swift on
Accelerate. Nothing was re-authored or retrained. Held to upstream's Python on a real piano
loop and a 17-second, three-segment piece:

| stage | result |
|---|---|
| trunk through Core AI (GPU) | 155 dB PSNR against upstream's activations |
| Swift GRU on upstream's own trunk activations | 112 dB against PyTorch |
| the seven framewise outputs | 123–147 dB |
| **note and pedal events** | **identical: every note, velocity and pedal, times within 0.001 ms** |

The post-processor is a line-for-line port, including a Python quirk (a note starting on the
very first frame is never emitted), because identical output is the point.

**What it scores, and what it is for.** On 540 of the General MIDI piano renders above (exact
truth), the piano engine reaches note F1 **0.87** (bass lines 0.92, pads 0.79) where MuScriptor
medium reaches 0.95 on the same files. That is the model, not the port — its events are
upstream's — and it is what a specialist trained on real piano recordings does with a synthesised
piano. On real piano its published number is 0.968 on MAESTRO. Its velocities correlate 0.53
with the MIDI velocities of the renders and sit about 20 below them; a GM synthesiser's dynamics
are not a Disklavier's, so treat that as a floor. Use it when you need velocity and pedal from
a piano recording; use MuScriptor for everything else, including synthesised piano.

## The model is not bundled

The weights are **CC BY-NC 4.0** (non-commercial) and gated on Hugging Face; the code is MIT.
The conversions described above are published, under the same licence and gate, at
[huggingface.co/arraypress/scribe-muscriptor](https://huggingface.co/arraypress/scribe-muscriptor)
(all three sizes, plus large fp16) and the tracker at
[huggingface.co/arraypress/scribe-beat-this](https://huggingface.co/arraypress/scribe-beat-this) (MIT):

```sh
# accept the licence on the repo page, then
hf download arraypress/scribe-muscriptor --include "scribe-medium-float32.aimodel/*" --local-dir models
hf download arraypress/scribe-beat-this --local-dir models
```

Or convert yourself, which is what produced those files:

```sh
# accept the licence at https://huggingface.co/MuScriptor, then
uvx hf auth login
git clone https://github.com/muscriptor/muscriptor ../muscriptor
git clone https://github.com/apple/coreai-models ../coreai-models
uv run Tools/export.py --size medium --install      # → ~/Library/Application Support/scribe/models/
```

`Tools/README.md` describes the export: one `.aimodel` carrying `main` (the decoder, with its
attention cache as Core AI state), `prefix` (mel and instrument conditioning) and `embed`
(tokens). `ModelLocator` finds assets by explicit path, `$SCRIBE_MODEL`, then the install directory.

## Options

`TranscriptionOptions` carries every knob upstream's `transcribe()` has, with the same defaults:
greedy decoding, or `sampling` with a `temperature` and optional `seed`; classifier-free
`guidance`; `instruments`, a hard mask so nothing else can be decoded; `preludeForcing`;
`beamSize`; `strictEndOfChunk`; the 2,000-token `maximumTokens` budget per chunk. Batching above
1 is accepted for its semantics (it disables forcing) but chunks are decoded one at a time.

`MIDIAssembly` writes the file upstream writes: one named track per instrument on channels 1–9
and 11–16, drums on 10, tempo and time signature from the grid, a `muscriptor:bar_offset=` marker
when the music was shifted to put bar 1 on a downbeat. `quantize` snaps to the detected
subdivision — what a score needs, not what anyone wants to hear.

`SheetMusic` engraves MusicXML and PDFs, with tablature for fretted parts, through MuseScore 4+
if it is installed. `Auralizer` renders a stereo check mix — original left, MIDI right — with
the system's General MIDI synthesiser or a SoundFont, no FluidSynth needed.

## Requirements

macOS 27, Apple silicon, Swift 6.2. Core AI and MusicUnderstanding are the floor; there is no
fallback for either.

## Tests

```sh
swift test                                           # 26 tests against golden fixtures, no model needed
SCRIBE_MODEL=~/Library/Application\ Support/scribe/models swift test   # + the end-to-end parity check
```

## License

MIT — see [LICENSE](LICENSE). The model weights keep their own licence (CC BY-NC 4.0).
