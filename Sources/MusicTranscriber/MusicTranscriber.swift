//
//  MusicTranscriber.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Audio in, notes out — MuScriptor on Core AI.
//
//  ```swift
//  let transcriber = try await MusicTranscriber(model: try ModelLocator.resolve())
//  let result = try await transcriber.transcribe(url)
//  try MIDIAssembly.data(notes: result.notes, grid: result.beatGrid).write(to: midiURL)
//  ```
//

import Foundation
import MusicAnalysis

/// A loaded transcription model.
///
/// One instance holds one model and can transcribe any number of files. Calls
/// are serialised: the attention cache is per-chunk state that one decode owns
/// at a time.
public final class MusicTranscriber: @unchecked Sendable {

    /// The decoder in use.
    public let decoder: CoreAIDecoder

    /// Seconds of audio per model chunk.
    public static let chunkSeconds = 5.0

    /// A hook for parity work: called with each chunk's forced prompt and the
    /// tokens the model generated for it. Nil in normal use.
    public var tokenObserver: (@Sendable (_ chunk: Int, _ prompt: [Int], _ tokens: [Int]) -> Void)?

    /// The Beat This! tracker, loaded once per instance on first use. Loading
    /// it per file cost a second a loop across a folder.
    private var beatTracker: BeatThisTracker?
    private var beatTrackerMissing = false

    /// Load a model asset. See ``ModelLocator``.
    public init(model url: URL) async throws {
        decoder = try await CoreAIDecoder(contentsOf: url)
    }

    // MARK: - Transcribing

    /// Transcribe a file, in full.
    ///
    /// - Parameters:
    ///   - url: any audio file AVFoundation decodes.
    ///   - options: how to decode; see ``TranscriptionOptions``.
    ///   - tempo: whether to detect a beat grid for the MIDI; see ``TempoDetection``.
    ///   - fixedTempo: a known BPM — a loop's, from its name — which skips the
    ///     tracker and writes that tempo in 4/4 with the downbeat at zero.
    ///   - tempoRange: where the tempo plausibly lives. A detection outside it is
    ///     snapped in by the smallest musical ratio (2, 1/2, 3/2, 2/3 …), which is
    ///     how a half-time hearing of a 178 BPM track becomes 178. See
    ///     ``BeatGrid/snapped(into:)``.
    ///   - tracker: which beat tracker finds the grid; see ``BeatTracker``.
    ///   - progress: called with the fraction of chunks done, on the caller's task.
    public func transcribe(_ url: URL, options: TranscriptionOptions = .init(),
                           tempo: TempoDetection = .bestEffort, fixedTempo: Double? = nil,
                           tempoRange: ClosedRange<Double>? = nil, tracker: BeatTracker = .automatic,
                           progress: ((Double) -> Void)? = nil) async throws -> Transcription {
        try options.validate()
        let samples = try AudioLoader.load(url)
        let audioDuration = Double(samples.count) / Double(MelSpectrogram.sampleRate)
        var warnings: [String] = []

        var grid: BeatGrid?
        if let fixedTempo {
            guard fixedTempo > 0 else { throw MusicTranscriberError.invalidOptions("a fixed tempo must be positive") }
            grid = .fixed(bpm: fixedTempo, duration: audioDuration)
        } else if tempo != .off {
            do {
                grid = try await detectGrid(url: url, samples: samples, duration: audioDuration, tracker: tracker)
            } catch let error as MusicTranscriberError {
                if tempo == .required { throw error }
                warnings.append("\(error.localizedDescription); falling back to the placeholder tempo")
            }
        }

        let started = Date()
        var records: [TranscriptionEvent.NoteEventRecord] = []
        let tally = Tally()
        for try await event in stream(samples: samples, options: options, tally: tally) {
            if let record = event.record { records.append(record) }
            if case .progress(let p) = event, p.total > 0 { progress?(Double(p.completed) / Double(p.total)) }
        }
        let decodeSeconds = Date().timeIntervalSince(started)
        warnings += tally.warnings
        let tokenCount = tally.tokens

        let events = records.map { record -> TranscriptionEvent in
            switch record { case .start(let s): return .noteStart(s); case .end(let e): return .noteEnd(e) }
        }
        let notes = NoteCleanup.cleaned(NoteCleanup.notes(from: events))
        if let tempoRange, let g = grid {
            // Drum hits are the strongest tempo evidence there is; fall back to
            // every onset when the track has few.
            let drums = notes.filter(\.isDrum).map(\.onset)
            grid = g.snapped(into: tempoRange, onsets: drums.count >= 20 ? drums : notes.map(\.onset))
        }
        if let g = grid { grid = g.withOnsetDelay(onsets: notes.map(\.onset)) }
        return Transcription(notes: notes, events: records, beatGrid: grid, warnings: warnings,
                             audioDuration: audioDuration, decodeSeconds: decodeSeconds, tokenCount: tokenCount)
    }

    /// Transcribe a file as a stream of events, chunk by chunk.
    ///
    /// The guarantees are upstream's: every note start is followed by its end,
    /// chunks arrive in order, and progress anchors are interleaved. Notes may all
    /// carry the same lag of up to ~25 ms; the beat grid removes it, and only
    /// exists once the stream has finished, so prefer ``transcribe(_:options:tempo:)``
    /// when timing matters.
    public func events(_ url: URL, options: TranscriptionOptions = .init()) throws -> AsyncThrowingStream<TranscriptionEvent, Error> {
        try options.validate()
        let samples = try AudioLoader.load(url)
        return stream(samples: samples, options: options, tally: Tally())
    }

    /// Counters a decode fills in from its own task.
    final class Tally: @unchecked Sendable {
        private let lock = NSLock()
        private var _tokens = 0
        private var _warnings: [String] = []
        var tokens: Int { lock.withLock { _tokens } }
        var warnings: [String] { lock.withLock { _warnings } }
        func add(tokens n: Int) { lock.withLock { _tokens += n } }
        func warn(_ message: String) { lock.withLock { _warnings.append(message) } }
    }

    // MARK: - The pipeline

    func stream(samples: [Float], options: TranscriptionOptions, tally: Tally) -> AsyncThrowingStream<TranscriptionEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(samples: samples, options: options, tally: tally) {
                        continuation.yield($0)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(samples: [Float], options: TranscriptionOptions, tally: Tally,
                     yield: (TranscriptionEvent) -> Void) async throws {
        let chunks = AudioLoader.chunks(samples)
        let mel = MelSpectrogram()
        let forcing = options.preludeForcing && options.batchSize <= 1
        let instrumentTokens: [Int32] = options.instruments.isEmpty ? [0] : options.instruments.map { Int32($0.id + 1) }
        let forbidden = options.instruments.isEmpty ? [] : EventVocabulary.forbiddenTokens(allowing: options.instruments)
        let generator = ChunkGenerator(decoder: decoder, options: options, forbidden: forbidden, instrumentTokens: instrumentTokens)
        let initialToken = decoder.card
        var rng: SeededGenerator? = options.seed.map(SeededGenerator.init(seed:))

        // One cache per beam, one more per beam under guidance, plus scratch for reordering.
        let beams = max(1, options.beamSize)
        let guided = options.guidance != 1.0
        let stateCount = beams > 1 ? beams * (guided ? 2 : 1) + beams * (guided ? 2 : 1) : (guided ? 2 : 1)
        let states = (0..<stateCount).map { _ in decoder.makeState() }

        var tokenDecoder = TokenDecoder()
        yield(.progress(.init(completed: 0, total: chunks.count)))

        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            let seekTime = Double(index) * Self.chunkSeconds
            let nextSeekTime = index + 1 < chunks.count ? Double(index + 1) * Self.chunkSeconds : nil

            // The boundary settles the tracker (a chunk that never emitted `tie`
            // drops its open notes) before the prompt is read from it.
            for event in tokenDecoder.beginChunk(seekTime: seekTime, nextSeekTime: nextSeekTime) { yield(event) }
            let prompt = forcing && index > 0 ? EventVocabulary.tiePrologue(for: tokenDecoder.openKeys) : []

            let conditioning = try await generator.conditioning(mel: mel.logMel(Array(chunk)))
            let result = try await generator.generate(conditioning: conditioning, prompt: prompt,
                                                      initialToken: initialToken, states: states, generator: &rng)
            tally.add(tokens: result.tokens.count)
            tokenObserver?(index, prompt, result.tokens)
            if !result.ended {
                let message = "chunk \(index) (seek=\(String(format: "%.1f", seekTime))s) did not emit EOS within \(options.maximumTokens) tokens"
                if options.strictEndOfChunk {
                    throw MusicTranscriberError.chunkDidNotEnd(chunk: index, seekTime: seekTime, budget: options.maximumTokens)
                }
                tally.warn(message)
            }
            for token in prompt + result.tokens {
                for event in tokenDecoder.feed(token) { yield(event) }
            }
            yield(.progress(.init(completed: index + 1, total: chunks.count)))
        }
        for event in tokenDecoder.finish() { yield(event) }
    }

    // MARK: - Tempo

    /// Detect the beat grid — Beat This! when installed or asked for, else
    /// MusicUnderstanding — and fit it with upstream's rules.
    public func detectGrid(url: URL, samples: [Float], duration: Double,
                           tracker: BeatTracker = .automatic) async throws -> BeatGrid {
        guard duration >= 1 else {
            throw MusicTranscriberError.noSteadyTempo(String(format: "Audio is %.2fs long, too short to detect a tempo", duration))
        }
        if tracker != .apple, let beatThis = try await loadBeatTracker(required: tracker == .beatThis) {
            let (beats, downbeats) = try await beatThis.track(samples16k: samples)
            return try BeatGridMath.grid(beats: beats, downbeats: downbeats)
        }
        return try await Self.detectGrid(url: url, duration: duration)
    }

    /// The cached tracker, loading it on first use; nil when it is not installed
    /// and not required.
    private func loadBeatTracker(required: Bool) async throws -> BeatThisTracker? {
        if let beatTracker { return beatTracker }
        if beatTrackerMissing && !required { return nil }
        do {
            let tracker = try await BeatThisTracker(contentsOf: try ModelLocator.resolveBeatTracker())
            beatTracker = tracker
            return tracker
        } catch MusicTranscriberError.modelNotFound where !required {
            beatTrackerMissing = true
            return nil
        }
    }

    /// Detect the beat grid with MusicUnderstanding and upstream's fitting rules.
    public static func detectGrid(url: URL, duration: Double) async throws -> BeatGrid {
        let analysis: AudioAnalysis
        do {
            analysis = try await MusicAnalysis.analyze(url: url, only: [.rhythm])
        } catch {
            throw MusicTranscriberError.noSteadyTempo("beat tracking failed: \(error.localizedDescription)")
        }
        guard let rhythm = analysis.rhythm else {
            throw MusicTranscriberError.noSteadyTempo("no beats detected")
        }
        return try BeatGridMath.grid(beats: rhythm.beats, downbeats: rhythm.bars)
    }
}

