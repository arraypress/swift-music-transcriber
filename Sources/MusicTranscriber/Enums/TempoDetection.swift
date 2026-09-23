//
//  TempoDetection.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// What to do about tempo when writing MIDI.
///
/// Upstream spells this `true | false | "best-effort"`. A failed detection
/// means the recording has no steady tempo or is too short — rubato piano,
/// most live recordings — and a wrong tempo in a MIDI file is worse than the
/// placeholder, so "best effort" warns and writes 120 BPM with no time signature.
public enum TempoDetection: String, Codable, Sendable, CaseIterable {

    /// Detect, and fail the transcription if no steady tempo is found.
    case required

    /// Detect, and fall back to the placeholder grid with a warning.
    case bestEffort

    /// Do not run the beat tracker at all.
    case off
}

/// Which beat tracker finds the grid.
///
/// Upstream uses Beat This! (CPJKU); this library ships it as a Core AI asset
/// and uses it whenever it is installed, because the grid then matches
/// upstream's. Apple's MusicUnderstanding needs no asset and remains the
/// fallback; measured on the demo it heard the half-time pulse (77.5 for 155).
public enum BeatTracker: String, Codable, Sendable, CaseIterable {

    /// Beat This! when installed, otherwise Apple's.
    case automatic

    /// Beat This!; an error when it is not installed.
    case beatThis = "beat-this"

    /// Apple's MusicUnderstanding.
    case apple
}
