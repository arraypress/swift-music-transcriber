//
//  TranscriptionEvent.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  What the streaming API yields while a transcription is running.
//

import Foundation

/// One item in the event stream.
///
/// Every ``noteStart(_:)`` is followed, later in the stream, by exactly one
/// ``noteEnd(_:)`` carrying the same `index`. Within a 5-second chunk events are
/// in time order, and every event of chunk N precedes every event of chunk N+1 —
/// the same guarantees upstream's Python generator makes. ``progress(_:)``
/// anchors are advisory: one up front with `completed == 0`, then one per chunk.
public enum TranscriptionEvent: Sendable, Hashable {

    /// A note began sounding.
    public struct NoteStart: Sendable, Hashable, Codable {
        /// MIDI note number.
        public let pitch: Int
        /// Seconds from the start of the audio.
        public let startTime: Double
        /// Stream-unique index, minted in order of appearance.
        public let index: Int
        /// Instrument group name, or `program_<n>`.
        public let instrument: String
        /// The program the model emitted, 128 for drums. Not serialised; upstream's JSON has no such field.
        public let program: Int
        /// Whether this is a drum hit.
        public let isDrum: Bool

        /// MIDI velocity when the engine measured one (the piano engine); absent for MuScriptor.
        public let velocity: Int?

        public init(pitch: Int, startTime: Double, index: Int, instrument: String, program: Int, isDrum: Bool, velocity: Int? = nil) {
            self.velocity = velocity
            self.pitch = pitch; self.startTime = startTime; self.index = index
            self.instrument = instrument; self.program = program; self.isDrum = isDrum
        }

        enum CodingKeys: String, CodingKey { case pitch, startTime = "start_time", index, instrument, program, isDrum = "is_drum", velocity }
    }

    /// A note stopped sounding.
    public struct NoteEnd: Sendable, Hashable, Codable {
        /// Seconds from the start of the audio.
        public let endTime: Double
        /// The `index` of the ``NoteStart`` this closes.
        public let startEventIndex: Int

        public init(endTime: Double, startEventIndex: Int) {
            self.endTime = endTime; self.startEventIndex = startEventIndex
        }

        enum CodingKeys: String, CodingKey { case endTime = "end_time", startEventIndex = "start_event_index" }
    }

    /// Coarse progress: `completed` of `total` chunks are done.
    public struct Progress: Sendable, Hashable, Codable {
        public let completed: Int
        public let total: Int
        public init(completed: Int, total: Int) { self.completed = completed; self.total = total }
    }

    case noteStart(NoteStart)
    case noteEnd(NoteEnd)
    case progress(Progress)
}
