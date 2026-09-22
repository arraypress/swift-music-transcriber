//
//  Transcription.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  What a finished transcription is.
//

import Foundation

/// A complete transcription: the notes, the grid they were written against,
/// and what the run cost.
public struct Transcription: Codable, Sendable {

    /// The cleaned notes, in onset order.
    public let notes: [TranscribedNote]

    /// The event stream the notes were built from, for callers that want the
    /// raw model output — upstream's `--format json`.
    public let events: [TranscriptionEvent.NoteEventRecord]

    /// The beat grid, or nil when tempo detection was off or found nothing.
    public let beatGrid: BeatGrid?

    /// Warnings worth showing: chunks that ran out of budget, a tempo that
    /// could not be detected under best effort.
    public let warnings: [String]

    /// Seconds of audio.
    public let audioDuration: Double

    /// Seconds of wall clock spent decoding.
    public let decodeSeconds: Double

    /// Tokens the model generated across every chunk.
    public let tokenCount: Int

    public init(notes: [TranscribedNote], events: [TranscriptionEvent.NoteEventRecord], beatGrid: BeatGrid?,
                warnings: [String], audioDuration: Double, decodeSeconds: Double, tokenCount: Int) {
        self.notes = notes
        self.events = events
        self.beatGrid = beatGrid
        self.warnings = warnings
        self.audioDuration = audioDuration
        self.decodeSeconds = decodeSeconds
        self.tokenCount = tokenCount
    }

    /// The instrument groups present, in order of first appearance.
    public var instruments: [String] {
        var seen: Set<String> = []
        return notes.compactMap { seen.insert($0.instrument).inserted ? $0.instrument : nil }
    }
}

extension TranscriptionEvent {

    /// A note event as upstream's JSON spells it: `{"type": "start", ...}` or `{"type": "end", ...}`.
    public enum NoteEventRecord: Codable, Sendable, Hashable {
        case start(NoteStart)
        case end(NoteEnd)

        private enum CodingKeys: String, CodingKey { case type }

        public init(from decoder: Decoder) throws {
            let type = try decoder.container(keyedBy: CodingKeys.self).decode(String.self, forKey: .type)
            switch type {
            case "start": self = .start(try NoteStart(from: decoder))
            case "end": self = .end(try NoteEnd(from: decoder))
            default: throw DecodingError.dataCorruptedError(forKey: .type, in: try decoder.container(keyedBy: CodingKeys.self),
                                                            debugDescription: "unknown event type \(type)")
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .start(let s):
                try container.encode("start", forKey: .type)
                try s.encode(to: encoder)
            case .end(let e):
                try container.encode("end", forKey: .type)
                try e.encode(to: encoder)
            }
        }
    }

    /// This event as a record, or nil for progress.
    public var record: NoteEventRecord? {
        switch self {
        case .noteStart(let s): return .start(s)
        case .noteEnd(let e): return .end(e)
        case .progress: return nil
        }
    }
}
