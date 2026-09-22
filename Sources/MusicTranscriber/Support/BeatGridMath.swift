//
//  BeatGridMath.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The tempo, meter and onset-alignment arithmetic from muscriptor.utils.beats.
//
//  Every constant here was tuned upstream against their test set and is kept
//  verbatim. The beat tracker itself is swapped — MusicUnderstanding instead of
//  beat_this — so the accuracy of *this* combination is what the README
//  should report, not theirs.
//

import Foundation

/// Pure functions over beat times and note onsets.
public enum BeatGridMath {

    /// Beats deviating more than this fraction of a beat, RMS, from a straight
    /// line means "no constant tempo".
    public static let maximumTempoResidual = 0.05

    /// Fraction of bars that must agree on a count to write a time signature.
    public static let minimumMeterAgreement = 0.9

    /// Fewer tracked beats than this is not a tempo.
    public static let minimumBeats = 8

    /// Candidate grids: binary and triplet divisions of the beat, simplest first.
    public static let onsetSubdivisions = [1, 2, 3, 4, 6, 8, 12, 16, 24]

    /// Resultant length below which the onsets sit on no grid.
    public static let minimumOnsetConcentration = 0.5

    /// Distinct onsets needed before a delay is believed.
    public static let minimumOnsets = 40

    /// The largest delay this is meant to correct, in seconds.
    public static let maximumOnsetDelay = 0.04

    /// Marker text recording the bar-alignment shift, so a renderer can line the
    /// synthesis back up with the original audio.
    public static let barOffsetMarker = "muscriptor:bar_offset="

    /// A measured lag of the onsets against a beat subdivision.
    public struct OnsetDelay: Equatable, Sendable {
        /// Signed seconds, positive when the onsets are late.
        public let seconds: Double
        /// Resultant length in 0…1: how tightly the onsets sit on the grid.
        public let concentration: Double
        /// Subdivisions per beat of the grid measured against.
        public let subdivision: Int
        /// Distinct onset times that went into it.
        public let onsetCount: Int
    }

    /// Least-squares tempo over a beat sequence: `(bpm, residual RMS seconds)`.
    ///
    /// A line through beat index against time beats the median interval: the
    /// tracker quantises beats to a frame grid, which alone limits a median-
    /// interval tempo to a few BPM of resolution.
    public static func fitTempo(_ beats: [Double]) -> (bpm: Double, residual: Double) {
        let n = Double(beats.count)
        let meanX = (n - 1) / 2
        let meanY = beats.reduce(0, +) / n
        var sxx = 0.0, sxy = 0.0
        for (i, y) in beats.enumerated() {
            let dx = Double(i) - meanX
            sxx += dx * dx
            sxy += dx * (y - meanY)
        }
        let slope = sxy / sxx
        let intercept = meanY - slope * meanX
        var sq = 0.0
        for (i, y) in beats.enumerated() {
            let r = y - (intercept + slope * Double(i))
            sq += r * r
        }
        return (60 / slope, (sq / n).squareRoot())
    }

    /// Beats per bar from downbeat spacing, or nil if the bars disagree.
    ///
    /// Only measures how far apart the downbeats are; it cannot tell whether
    /// they are on the right beat. A tracker reporting two beats per bar for 3/4
    /// is self-consistent here, which is why the agreement bar is high.
    public static func inferBeatsPerBar(beats: [Double], downbeats: [Double],
                                        minimumAgreement: Double = minimumMeterAgreement) -> Int? {
        guard downbeats.count >= 3, beats.count >= 2 else { return nil }
        let beat = median(zip(beats.dropFirst(), beats).map { $0 - $1 })
        let counts = zip(downbeats.dropFirst(), downbeats)
            .map { Int((($0 - $1) / beat).rounded(.toNearestOrEven)) }
            .filter { $0 >= 2 }
        guard !counts.isEmpty else { return nil }
        var tally: [Int: Int] = [:]
        for c in counts { tally[c, default: 0] += 1 }
        // np.unique sorts ascending and argmax takes the first maximum.
        let best = tally.sorted { $0.key < $1.key }.max { a, b in a.value < b.value || (a.value == b.value && a.key > b.key) }!
        guard Double(best.value) / Double(counts.count) >= minimumAgreement else { return nil }
        return best.key
    }

    /// Onset times as positions in continuous beats, one entry per distinct
    /// millisecond, onsets outside the tracked span dropped.
    public static func onsetPhases(onsets: [Double], beats: [Double]) -> [Double] {
        let times = Array(Set(onsets.map { ($0 * 1000).rounded(.toNearestOrEven) / 1000 })).sorted()
        guard let first = beats.first, let last = beats.last else { return [] }
        return times.filter { $0 >= first && $0 <= last }.map { t in
            // np.interp: linear between the two beats around t.
            var hi = beats.firstIndex { $0 >= t } ?? beats.count - 1
            if hi == 0 { return 0 }
            if hi >= beats.count { hi = beats.count - 1 }
            let lo = hi - 1
            let span = beats[hi] - beats[lo]
            return span == 0 ? Double(hi) : Double(lo) + (t - beats[lo]) / span
        }
    }

    /// Mean resultant of the phases on a `subdivision` grid: `(concentration, offset in beats)`.
    public static func phaseConcentration(_ phases: [Double], subdivision: Int) -> (concentration: Double, beats: Double) {
        var re = 0.0, im = 0.0
        for p in phases {
            let angle = 2 * Double.pi * ((p * Double(subdivision)).truncatingRemainder(dividingBy: 1) + 1).truncatingRemainder(dividingBy: 1)
            re += cos(angle); im += sin(angle)
        }
        re /= Double(phases.count); im /= Double(phases.count)
        let turns = 1 / (2 * Double.pi * Double(subdivision))
        return ((re * re + im * im).squareRoot(), atan2(im, re) * turns)
    }

    /// How late `onsets` sit against the beat subdivision they are on, or nil.
    ///
    /// Each onset is a unit vector on a circle whose angle is its position in the
    /// beat; the mean vector's angle is the average offset and its length says how
    /// well they align. Repeated for each candidate subdivision; the tightest wins.
    public static func estimateOnsetDelay(onsets: [Double], beats: [Double], bpm: Double) -> OnsetDelay? {
        guard beats.count >= 2 else { return nil }
        let period = 60 / bpm
        let phases = onsetPhases(onsets: onsets, beats: beats)
        guard phases.count >= minimumOnsets else { return nil }
        let candidates = onsetSubdivisions.filter { period / (2 * Double($0)) >= maximumOnsetDelay }
        guard !candidates.isEmpty else { return nil }
        let scored = candidates.map { ($0, phaseConcentration(phases, subdivision: $0)) }
        let best = scored.max { $0.1.concentration < $1.1.concentration }!
        guard best.1.concentration >= minimumOnsetConcentration else { return nil }
        let seconds = best.1.beats * period
        guard abs(seconds) <= maximumOnsetDelay else { return nil }
        return OnsetDelay(seconds: seconds, concentration: best.1.concentration,
                          subdivision: best.0, onsetCount: phases.count)
    }

    /// Fit a grid from tracked beats and downbeats, upstream's `detect_grid`
    /// minus the tracker.
    ///
    /// Throws when there are too few beats or they do not fit a constant tempo.
    /// An unclear meter is not fatal: the grid comes back with no `beatsPerBar`.
    public static func grid(beats: [Double], downbeats: [Double]) throws -> BeatGrid {
        guard beats.count >= minimumBeats else {
            throw MusicTranscriberError.noSteadyTempo("Only \(beats.count) beats detected, need at least \(minimumBeats)")
        }
        let (bpm, residual) = fitTempo(beats)
        let beatSeconds = 60 / bpm
        guard residual <= maximumTempoResidual * beatSeconds else {
            throw MusicTranscriberError.noSteadyTempo(String(
                format: "The recording has no fixed tempo (beats deviate %.0f ms RMS from a constant %.1f BPM)",
                residual * 1000, bpm))
        }
        return BeatGrid(bpm: bpm,
                        beatsPerBar: inferBeatsPerBar(beats: beats, downbeats: downbeats),
                        firstDownbeat: downbeats.first ?? beats.first ?? 0,
                        beats: beats)
    }

    /// Notes moved by `delay` seconds. May go negative; the bar offset covers that.
    public static func shifted(_ notes: [TranscribedNote], by delay: Double) -> [TranscribedNote] {
        guard delay != 0 else { return notes }
        return notes.map { var n = $0; n.onset += delay; n.offset += delay; return n }
    }

    /// Every onset and offset moved onto a `step`-second grid, in the timeline
    /// the MIDI file will have (`offset` is the shift that puts bar 1 at tick 0).
    ///
    /// A note shorter than half a step would vanish from the score, so it gets
    /// the shortest length the grid has; that can make two notes of one pitch
    /// overlap, hence the trim.
    public static func quantized(_ notes: [TranscribedNote], step: Double, offset: Double) -> [TranscribedNote] {
        func snap(_ t: Double) -> Double { ((t + offset) / step).rounded(.toNearestOrEven) * step - offset }
        let snapped = notes.map { note -> TranscribedNote in
            var n = note
            n.onset = snap(note.onset)
            n.offset = max(snap(note.offset), n.onset + step)
            return n
        }
        return NoteCleanup.cleaned(snapped)
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
    }
}
