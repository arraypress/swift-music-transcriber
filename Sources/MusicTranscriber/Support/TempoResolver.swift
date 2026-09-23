//
//  TempoResolver.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The beat grid for a recording, shared by both engines: a fixed tempo, or
//  Beat This! when its asset is installed and asked for, else Apple's
//  MusicUnderstanding. The tracker is loaded once and kept; a missing asset is
//  remembered so a folder of loops does not look for it a hundred times.
//

import Foundation

/// Resolves and caches the beat tracker, and turns the tempo options into a grid.
public final class TempoResolver: @unchecked Sendable {

    private var beatTracker: BeatThisTracker?
    private var beatTrackerMissing = false

    public init() {}

    /// The grid the options ask for, or nil when tempo detection is off or
    /// failed in best-effort mode (with a warning appended).
    public func grid(url: URL, samples: [Float], duration: Double, tempo: TempoDetection, fixedTempo: Double?,
                     tracker: BeatTracker, warnings: inout [String]) async throws -> BeatGrid? {
        if let fixedTempo {
            guard fixedTempo > 0 else { throw MusicTranscriberError.invalidOptions("a fixed tempo must be positive") }
            return .fixed(bpm: fixedTempo, duration: duration)
        }
        guard tempo != .off else { return nil }
        do {
            return try await detect(url: url, samples: samples, duration: duration, tracker: tracker)
        } catch let error as MusicTranscriberError {
            if tempo == .required { throw error }
            warnings.append("\(error.localizedDescription); falling back to the placeholder tempo")
            return nil
        }
    }

    /// Detect the grid with the chosen tracker.
    public func detect(url: URL, samples: [Float], duration: Double, tracker: BeatTracker = .automatic) async throws -> BeatGrid {
        guard duration >= 1 else {
            throw MusicTranscriberError.noSteadyTempo(String(format: "Audio is %.2fs long, too short to detect a tempo", duration))
        }
        if tracker != .apple, let beatThis = try await loadBeatTracker(required: tracker == .beatThis) {
            let (beats, downbeats) = try await beatThis.track(samples16k: samples)
            return try BeatGridMath.grid(beats: beats, downbeats: downbeats)
        }
        return try await MusicTranscriber.detectGrid(url: url, duration: duration)
    }

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
}
