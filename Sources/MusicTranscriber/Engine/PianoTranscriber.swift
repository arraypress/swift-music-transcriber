//
//  PianoTranscriber.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The piano specialist: Kong, Li, Song and Wang's high-resolution piano
//  transcription with pedals (ByteDance, 2020; Apache 2.0), the model that
//  gives velocity and sustain, which MuScriptor cannot. The convolutional
//  trunks run in the .aimodel; the sixteen bidirectional GRUs and their heads
//  run here from weights the asset carries, because Core AI has no recurrent
//  op. Segmentation, stitching and post-processing are upstream's.
//

import CoreAI
import Foundation

/// Piano notes with velocity, and sustain-pedal events, from 16 kHz audio.
public final class PianoTranscriber: @unchecked Sendable {

    /// The asset name the export writes and the installer expects.
    public static let assetName = "scribe-piano-float32.aimodel"

    /// Branch order of the `trunk` entry point's outputs, and of the parameter vector.
    static let branches = ["frame_model", "reg_onset_model", "reg_offset_model", "velocity_model",
                           "reg_pedal_onset_model", "reg_pedal_offset_model", "reg_pedal_frame_model"]
    static let hidden = 256
    static let trunkWidth = 768

    public let url: URL
    private let trunk: InferenceFunction
    private let frontEnd = PianoFrontEnd()
    /// Seven branch heads (two-layer GRU + linear), then the two conditioning heads.
    let heads: [GRUHead]
    let onsetConditioning: GRUHead
    let frameConditioning: GRUHead

    public init(contentsOf url: URL) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MusicTranscriberError.modelNotFound(url.path)
        }
        self.url = url
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model: AIModel
        do { model = try await AIModel(contentsOf: url, options: options) } catch {
            throw MusicTranscriberError.inferenceFailed("could not load \(url.lastPathComponent): \(error)")
        }
        guard let trunk = try model.loadFunction(named: "trunk"), let parameters = try model.loadFunction(named: "parameters") else {
            throw MusicTranscriberError.modelIncompatible("piano model needs trunk and parameters; found \(model.functionNames)")
        }
        self.trunk = trunk

        var outputs = try await parameters.run(inputs: ["probe": CoreAIDecoder.array([0], shape: [1], half: false)])
        guard let flat = outputs.remove("parameters")?.ndArray else {
            throw MusicTranscriberError.modelIncompatible("piano model returned no parameters")
        }
        let p = CoreAIDecoder.floats(flat)
        var cursor = 0
        var heads: [GRUHead] = []
        for (i, _) in Self.branches.enumerated() {
            heads.append(GRUHead.parse(p, cursor: &cursor, layers: 2, inputSize: Self.trunkWidth, hidden: Self.hidden, classes: i < 4 ? 88 : 1))
        }
        self.heads = heads
        onsetConditioning = GRUHead.parse(p, cursor: &cursor, layers: 1, inputSize: 88 * 2, hidden: Self.hidden, classes: 88)
        frameConditioning = GRUHead.parse(p, cursor: &cursor, layers: 1, inputSize: 88 * 3, hidden: Self.hidden, classes: 88)
        guard cursor == p.count else {
            throw MusicTranscriberError.modelIncompatible("piano parameters: read \(cursor) of \(p.count) floats")
        }
    }

    // MARK: - Whole recordings

    /// The beat tracker, shared with a ``MusicTranscriber`` when one is set here.
    public var tempoResolver = TempoResolver()

    /// Transcribe a file: notes with velocity on one acoustic-piano track,
    /// sustain-pedal events, and the same tempo handling as the MuScriptor path.
    public func transcribe(_ url: URL, tempo: TempoDetection = .bestEffort, fixedTempo: Double? = nil,
                           tempoRange: ClosedRange<Double>? = nil, tracker: BeatTracker = .automatic) async throws -> Transcription {
        let samples = try AudioLoader.load(url)
        let audioDuration = Double(samples.count) / Double(PianoFrontEnd.sampleRate)
        var warnings: [String] = []
        var grid = try await tempoResolver.grid(url: url, samples: samples, duration: audioDuration, tempo: tempo,
                                                fixedTempo: fixedTempo, tracker: tracker, warnings: &warnings)
        let started = Date()
        let (events, pedals) = try await transcribe(samples16k: samples)
        let decodeSeconds = Date().timeIntervalSince(started)

        // Upstream lists notes key by key; chronological order reads better and indexes the events by time.
        let notes = events.map {
            TranscribedNote(pitch: $0.midiNote, onset: Double($0.onset), offset: Double($0.offset),
                            instrument: InstrumentGroup.label(forProgram: 0), program: 0, isDrum: false, velocity: $0.velocity)
        }.sorted { ($0.onset, $0.pitch) < ($1.onset, $1.pitch) }
        // The event stream upstream's writer would see: starts and ends in time order.
        var timeline: [(time: Double, order: Int, record: TranscriptionEvent.NoteEventRecord)] = []
        for (i, n) in notes.enumerated() {
            timeline.append((n.onset, 1, .start(.init(pitch: n.pitch, startTime: n.onset, index: i, instrument: n.instrument,
                                                      program: n.program, isDrum: false, velocity: n.velocity))))
            timeline.append((n.offset, 0, .end(.init(endTime: n.offset, startEventIndex: i))))
        }
        let records = timeline.sorted { ($0.time, $0.order) < ($1.time, $1.order) }.map(\.record)

        if let tempoRange, let g = grid { grid = g.snapped(into: tempoRange, onsets: notes.map(\.onset)) }
        if let g = grid { grid = g.withOnsetDelay(onsets: notes.map(\.onset)) }
        return Transcription(notes: notes, events: records, beatGrid: grid, warnings: warnings,
                             audioDuration: audioDuration, decodeSeconds: decodeSeconds, tokenCount: 0,
                             pedals: pedals.map { PedalEvent(onset: Double($0.onset), offset: Double($0.offset)) })
    }

    /// Note and pedal events for a 16 kHz mono signal: upstream's `transcribe`.
    public func transcribe(samples16k: [Float]) async throws -> (notes: [PianoPostprocessing.NoteEvent], pedals: [PianoPostprocessing.PedalEvent]) {
        PianoPostprocessing.events(from: try await outputs(samples16k: samples16k))
    }

    /// Framewise outputs for a whole recording: ten-second segments at a
    /// five-second hop over the zero-padded signal, stitched as upstream does.
    public func outputs(samples16k: [Float]) async throws -> PianoPostprocessing.Outputs {
        let segment = PianoFrontEnd.segmentSamples
        let paddedCount = Int((Double(samples16k.count) / Double(segment)).rounded(.up)) * segment
        var padded = samples16k
        padded.append(contentsOf: [Float](repeating: 0, count: paddedCount - samples16k.count))
        var starts: [Int] = []
        var pointer = 0
        while pointer + segment <= padded.count { starts.append(pointer); pointer += segment / 2 }

        var perSegment: [[String: [Float]]] = []
        for start in starts {
            let mel = frontEnd.logMel(segment: Array(padded[start..<(start + segment)]))
            perSegment.append(try await segmentOutputs(mel: mel))
        }
        let frames = PianoFrontEnd.segmentFrames
        func stitch(_ key: String, width: Int) -> (values: [Float], frames: Int) {
            PianoPostprocessing.deframe(perSegment.map { $0[key]! }, segmentFrames: frames, width: width)
        }
        let regOnset = stitch("reg_onset_output", width: 88)
        return PianoPostprocessing.Outputs(frames: regOnset.frames,
                                           regOnset: regOnset.values,
                                           regOffset: stitch("reg_offset_output", width: 88).values,
                                           frame: stitch("frame_output", width: 88).values,
                                           velocity: stitch("velocity_output", width: 88).values,
                                           pedalOnset: stitch("reg_pedal_onset_output", width: 1).values,
                                           pedalOffset: stitch("reg_pedal_offset_output", width: 1).values,
                                           pedalFrame: stitch("pedal_frame_output", width: 1).values)
    }

    // MARK: - One segment

    /// The seven outputs upstream's `Note_pedal.forward` returns for one
    /// segment of log-mel (`[segmentFrames][mels]`), each `[segmentFrames][classes]`.
    public func segmentOutputs(mel: [Float]) async throws -> [String: [Float]] {
        let frames = PianoFrontEnd.segmentFrames
        let trunks = try await trunkFeatures(mel: mel)
        var branch: [String: [Float]] = [:]
        for (i, name) in Self.branches.enumerated() {
            branch[name] = heads[i].run(trunks[i], frames: frames)
        }
        // Velocities condition the onset regression: cat(onset, sqrt(onset) · velocity).
        let onset = branch["reg_onset_model"]!, velocity = branch["velocity_model"]!
        var joined = [Float](repeating: 0, count: frames * 176)
        for t in 0..<frames {
            for k in 0..<88 {
                joined[t * 176 + k] = onset[t * 88 + k]
                joined[t * 176 + 88 + k] = onset[t * 88 + k].squareRoot() * velocity[t * 88 + k]
            }
        }
        let regOnset = onsetConditioning.run(joined, frames: frames)
        // Onsets and offsets condition the frame-wise classification.
        let frame = branch["frame_model"]!, offset = branch["reg_offset_model"]!
        var joined3 = [Float](repeating: 0, count: frames * 264)
        for t in 0..<frames {
            for k in 0..<88 {
                joined3[t * 264 + k] = frame[t * 88 + k]
                joined3[t * 264 + 88 + k] = regOnset[t * 88 + k]
                joined3[t * 264 + 176 + k] = offset[t * 88 + k]
            }
        }
        let frameOutput = frameConditioning.run(joined3, frames: frames)
        return ["reg_onset_output": regOnset,
                "reg_offset_output": offset,
                "frame_output": frameOutput,
                "velocity_output": velocity,
                "reg_pedal_onset_output": branch["reg_pedal_onset_model"]!,
                "reg_pedal_offset_output": branch["reg_pedal_offset_model"]!,
                "pedal_frame_output": branch["reg_pedal_frame_model"]!]
    }

    /// The seven trunks' features for one segment, `[segmentFrames][768]` each, from the asset.
    public func trunkFeatures(mel: [Float]) async throws -> [[Float]] {
        let frames = PianoFrontEnd.segmentFrames
        precondition(mel.count == frames * PianoFrontEnd.mels)
        var outputs = try await trunk.run(inputs: ["mel": CoreAIDecoder.array(mel, shape: [1, frames, PianoFrontEnd.mels], half: false)])
        return try Self.branches.map { name in
            guard let a = outputs.remove(name)?.ndArray else {
                throw MusicTranscriberError.inferenceFailed("piano trunk produced no \(name)")
            }
            return CoreAIDecoder.floats(a)
        }
    }
}
