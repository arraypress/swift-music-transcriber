//
//  BeatThisTracker.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Beat This! on Core AI: the beat tracker MuScriptor uses, so the grid the
//  MIDI is written against can match upstream's rather than Apple's.
//

import CoreAI
import Foundation

/// The Beat This! beat and downbeat tracker (CPJKU, MIT), exported as one
/// `.aimodel` by `Tools/export_beat_this.py`: `spect [1,T,128]` in, framewise
/// `beat` and `downbeat` logits `[1,T]` out, 50 frames a second.
public final class BeatThisTracker: @unchecked Sendable {

    /// The asset file name this library looks for.
    public static let assetName = "scribe-beat-this-float32.aimodel"

    public let url: URL
    private let function: InferenceFunction
    private let frontEnd = BeatThisFrontEnd()

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
        guard let function = try model.loadFunction(named: "main") else {
            throw MusicTranscriberError.modelIncompatible("beat tracker has no main function; found \(model.functionNames)")
        }
        self.function = function
    }

    /// Beats and downbeats, in seconds, for the model's 16 kHz mono signal —
    /// resampled to 22.05 kHz here, as upstream does before its front end.
    public func track(samples16k: [Float]) async throws -> (beats: [Double], downbeats: [Double]) {
        let samples = Resampler(from: MelSpectrogram.sampleRate, to: BeatThisFrontEnd.sampleRate).resample(samples16k)
        let (mel, frames) = frontEnd.logMel(samples)
        let (beat, downbeat) = try await logits(mel: mel, frames: frames)
        return BeatPostprocessing.beats(beatLogits: beat, downbeatLogits: downbeat)
    }

    /// One pass of the model over `rows` frames of log-mel, no chunking, no border
    /// handling: what upstream's `model(chunk)` returns for that chunk.
    public func run(chunk: [Float], rows: Int) async throws -> (beat: [Float], downbeat: [Float]) {
        precondition(chunk.count == rows * BeatThisFrontEnd.mels)
        var outputs = try await function.run(inputs: ["spect": CoreAIDecoder.array(chunk, shape: [1, rows, BeatThisFrontEnd.mels], half: false)])
        guard let b = outputs.remove("beat")?.ndArray, let d = outputs.remove("downbeat")?.ndArray else {
            throw MusicTranscriberError.inferenceFailed("beat tracker produced no logits")
        }
        return (CoreAIDecoder.floats(b), CoreAIDecoder.floats(d))
    }

    /// Framewise logits for a whole piece, chunked and stitched as upstream does.
    public func logits(mel: [Float], frames: Int) async throws -> (beat: [Float], downbeat: [Float]) {
        let mels = BeatThisFrontEnd.mels
        var beatChunks: [(BeatPostprocessing.Chunk, [Float])] = []
        var downbeatChunks: [(BeatPostprocessing.Chunk, [Float])] = []
        for chunk in BeatPostprocessing.chunks(frames: frames) {
            var input = [Float](repeating: 0, count: chunk.padLeft * mels)
            input.append(contentsOf: mel[(chunk.from * mels)..<(chunk.to * mels)])
            input.append(contentsOf: [Float](repeating: 0, count: chunk.padRight * mels))
            let (b, d) = try await run(chunk: input, rows: input.count / mels)
            beatChunks.append((chunk, b))
            downbeatChunks.append((chunk, d))
        }
        return (BeatPostprocessing.aggregate(chunkLogits: beatChunks, frames: frames),
                BeatPostprocessing.aggregate(chunkLogits: downbeatChunks, frames: frames))
    }
}
