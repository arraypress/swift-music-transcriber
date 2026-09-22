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
