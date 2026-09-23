//
//  Note.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  One transcribed note: what the model heard, when, on which instrument.
//

import Foundation

/// A note the model transcribed.
///
/// Times are seconds from the start of the audio. MuScriptor's vocabulary has
/// one bit for note-on versus note-off and nothing for loudness, so its notes
/// carry the constant ``velocity`` of 100; the piano engine measures one.
public struct TranscribedNote: Hashable, Codable, Sendable {

    /// MIDI note number, 0…127.
    public let pitch: Int

    /// When it starts, in seconds.
    public var onset: Double

    /// When it stops, in seconds. For drums this is the onset plus 10 ms.
    public var offset: Double

    /// The instrument group the model assigned, e.g. `"acoustic_piano"`, or
    /// `program_<n>` for a program outside the table.
    public let instrument: String

    /// The General MIDI program the model emitted for this note, or 128 for drums.
    public let program: Int

    /// Whether it is a drum hit. Drums have no length; ``offset`` is nominal.
    public let isDrum: Bool

    /// MIDI velocity, 1…127. A constant 100 from MuScriptor, which has no
    /// loudness; measured by the piano engine.
    public var velocity: Int

    public init(pitch: Int, onset: Double, offset: Double, instrument: String, program: Int, isDrum: Bool, velocity: Int = 100) {
        self.pitch = pitch
        self.onset = onset
        self.offset = offset
        self.instrument = instrument
        self.program = program
        self.isDrum = isDrum
        self.velocity = min(127, max(1, velocity))
    }

    /// Length in seconds.
    public var duration: Double { offset - onset }

    /// The shortest note the decoder will write: 10 ms, upstream's floor.
    public static let minimumDuration = 0.01
}

/// A sustain-pedal press, from the piano engine: down at ``onset``, up at ``offset``, seconds.
public struct PedalEvent: Hashable, Codable, Sendable {
    public var onset: Double
    public var offset: Double
    public init(onset: Double, offset: Double) {
        self.onset = onset
        self.offset = offset
    }
}
