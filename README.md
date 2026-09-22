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
| log-mel front end (vDSP) | fixture from `torchaudio` | > 90 dB PSNR |
| sinc resampler (julius port) | fixtures at 44.1 and 48 kHz | > 90 dB PSNR |
| token decoder, tie prologue, note cleanup | upstream's event stream for 3 real chunks | identical, 346 events |
| MIDI layout | upstream's file for the same notes | same notes, ticks, tracks, tempo |
| **whole pipeline**, medium fp32, on the repo's demo clip | upstream's greedy CPU decode | **token for token, all 3 chunks** |
| large, fp32 and fp16 | same | token for token |
| small fp16 | same | diverges after ~50 tokens — where upstream's own fp16 run diverges |

So fp32 is the default. On this hardware it was no slower than fp16 for `medium`; for `large`,
which is bandwidth-bound at 5.5 GB, fp16 is twice as fast and still exact on the demo.

**The MP3 decoder is a bigger variable than the precision.** The parity rows above feed the
model the same 16 kHz samples upstream decoded with libsndfile. Given the MP3 itself,
AVFoundation decodes it slightly differently, and on `large` that moved the transcription from
190 notes to 161 (medium was unchanged). For reproducible results feed WAV or FLAC.

**The beat tracker is not upstream's.** Tempo and metre come from Apple's MusicUnderstanding
(via [swift-music-analysis](https://github.com/arraypress/swift-music-analysis)); upstream uses
`beat_this`. The fitting rules on top — least-squares tempo, 90% downbeat agreement for a metre,
circular-statistics onset delay — are ported verbatim. On the demo clip MusicUnderstanding
reports **77.5 BPM where beat_this reports 154.9**: the same grid an octave apart. Both are
defensible for that material; it is the one place the output can differ from upstream's
by design, and `TempoDetection.off` sidesteps it.

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

- **Synths are not a class the model reaches for.** Twenty synth loops, zero `synth_lead` or
  `synth_pad` labels; they come back as guitars and pianos. With
  `--instruments synth_lead,synth_pad,electric_bass,drums` the same loops produce **48 notes
  per file instead of 37**, labelled by construction. On electronic material, mask.
- **Untuned percussion barely registers.** Bongo and shaker loops yield one to four notes; kit
  loops yield twenty and are labelled drums. Vocals are transcribed as pitch (18 notes per loop)
  but rarely labelled `voice` (3 of 40).
- **Tempo holds where there is a beat.** Within 1 BPM on every kit loop and 31 of 40 bass loops;
  the 41 loops with no grid are mostly shakers, bongos and half the vocals, where a tracker has
  nothing to hold on to. Five loops came back an octave off.

## The model is not bundled

The weights are **CC BY-NC 4.0** (non-commercial) and gated on Hugging Face; the code is MIT.
Convert once:

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
