//
//  BeatPostprocessing.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Beat This!'s "minimal" post-processor and its chunking arithmetic, as pure
//  functions over frame logits.
//

import Foundation

/// From framewise logits to beat and downbeat times.
public enum BeatPostprocessing {

    /// Frames the model sees per pass; its training length.
    public static let chunkFrames = 1500

    /// Frames discarded at each edge of a pass, where the model was never
    /// trained (the loss max-pools over them).
    public static let borderFrames = 6

    /// One pass over the spectrogram: which frames go in, and how the input is
    /// zero-padded so the borders can be cut afterwards.
    public struct Chunk: Equatable {
        /// Start in the piece, may be negative.
        public let start: Int
        /// Rows of the spectrogram to take: `[from, to)`.
        public let from: Int
        public let to: Int
        /// Zero rows added before and after.
        public let padLeft: Int
        public let padRight: Int
    }

    /// Upstream's `split_piece`: chunks of 1,500 frames overlapping by twice the
    /// border, the first padded on the left, the last shifted so it ends at the
    /// end of the piece rather than running short.
    public static func chunks(frames: Int, chunk: Int = chunkFrames, border: Int = borderFrames) -> [Chunk] {
        let step = chunk - 2 * border
        var starts: [Int] = Array(stride(from: -border, to: frames - border, by: step))
        if starts.isEmpty { starts = [-border] }
        if frames > step { starts[starts.count - 1] = frames - (chunk - border) }
        return starts.map { start in
            let from = max(start, 0), to = min(start + chunk, frames)
            return Chunk(start: start, from: from, to: to,
                         padLeft: max(0, -start),
                         padRight: max(0, min(border, start + chunk - frames)))
        }
    }

    /// Upstream's `aggregate_prediction` with `keep_first`: each chunk's
    /// logits minus their borders, written into the piece; earlier chunks win
    /// where they overlap. Frames no chunk covered stay at −1000.
    public static func aggregate(chunkLogits: [(chunk: Chunk, logits: [Float])], frames: Int,
                                 chunk: Int = chunkFrames, border: Int = borderFrames) -> [Float] {
        var piece = [Float](repeating: -1000, count: frames)
        for (c, logits) in chunkLogits.reversed() {
            let inner = Array(logits[border..<(logits.count - border)])
            let base = c.start + border
            for (i, v) in inner.enumerated() {
                let index = base + i
                if index >= 0 && index < frames { piece[index] = v }
            }
        }
        return piece
    }

    /// Upstream's minimal post-processing: peaks of the logits within ±3 frames
    /// that are above 0 (probability one half), adjacent peaks averaged, then
    /// every downbeat moved onto its nearest beat and duplicates dropped.
    public static func beats(beatLogits: [Float], downbeatLogits: [Float], fps: Int = 50) -> (beats: [Double], downbeats: [Double]) {
        let beatFrames = deduplicate(peaks(beatLogits))
        var downbeatFrames = deduplicate(peaks(downbeatLogits))
        let beats = beatFrames.map { $0 / Double(fps) }
        var downbeats = downbeatFrames.map { $0 / Double(fps) }
        if !beats.isEmpty {
            downbeats = downbeats.map { d in beats.min { abs($0 - d) < abs($1 - d) }! }
            downbeats = Array(Set(downbeats)).sorted()
        }
        downbeatFrames = []
        return (beats, downbeats)
    }

    /// Indices where the logit equals its max over a 7-frame window and exceeds 0.
    static func peaks(_ logits: [Float]) -> [Int] {
        var out: [Int] = []
        for i in logits.indices where logits[i] > 0 {
            let lo = max(0, i - 3), hi = min(logits.count - 1, i + 3)
            var isMax = true
            for j in lo...hi where logits[j] > logits[i] { isMax = false; break }
            if isMax { out.append(i) }
        }
        return out
    }

    /// Groups of peaks not more than one frame apart become their mean.
    static func deduplicate(_ peaks: [Int]) -> [Double] {
        var result: [Double] = []
        guard var p = peaks.first.map(Double.init) else { return result }
        var c = 1.0
        for next in peaks.dropFirst() {
            let n = Double(next)
            if n - p <= 1 {
                c += 1
                p += (n - p) / c
            } else {
                result.append(p)
                p = n
                c = 1
            }
        }
        result.append(p)
        return result
    }
}
