//
//  BeatGrid.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  A constant-tempo grid to write the MIDI against.
//

import Foundation

/// A steady tempo detected from the audio, with the meter where the bars agreed.
///
/// Upstream fits this from a beat tracker's beats and downbeats; here the beats
/// come from Apple's MusicUnderstanding via `swift-music-analysis`, and the
/// arithmetic on top of them is ported unchanged from `muscriptor.utils.beats`.
public struct BeatGrid: Codable, Sendable, Equatable {

    /// Quarter notes per minute, least-squares over the tracked beats.
    public let bpm: Double

    /// Beats per bar, or nil when the downbeats did not agree on one — then no
    /// time signature is written, because a wrong one is worse than none.
    public let beatsPerBar: Int?

    /// Time of the first bar line, in seconds.
    public let firstDownbeat: Double

    /// The tracked beat times the grid was fitted to. Kept because the onset
    /// delay is measured against *these*, which follow the recording's small
    /// wobbles, not against the fitted tempo. Nil for a hand-built grid.
    public let beats: [Double]?

    /// Seconds by which the transcribed onsets sit late against the beats, once
    /// measured. Whoever writes the notes subtracts it. Nil until measured.
    public private(set) var onsetDelay: Double?

    /// Subdivisions of the beat the onsets were found to sit on, when they sat
    /// tightly enough on any; what ``quantize`` snaps to.
    public private(set) var beatSubdivision: Int?

    /// Whether the tempo was given rather than tracked. A given grid is never
    /// bar-shifted: its downbeat is the start of the file. The onset lag is
    /// still measured and removed — on a 123 BPM piano loop the model ran a
    /// steady 25 ms early against the sixteenth grid, and that is what makes a
    /// loop "not aligned" in a DAW.
    public let isFixed: Bool

    public init(bpm: Double, beatsPerBar: Int?, firstDownbeat: Double, beats: [Double]? = nil,
                onsetDelay: Double? = nil, beatSubdivision: Int? = nil, isFixed: Bool = false) {
        self.bpm = bpm
        self.beatsPerBar = beatsPerBar
        self.firstDownbeat = firstDownbeat
        self.beats = beats
        self.onsetDelay = onsetDelay
        self.beatSubdivision = beatSubdivision
        self.isFixed = isFixed
    }

    /// A grid at a known tempo — a loop's, from its filename — with the first
    /// downbeat at zero and a beat every `60 / bpm` seconds across `duration`.
    ///
    /// The synthetic beats let ``withOnsetDelay(onsets:)`` measure the lag and
    /// the subdivision the notes sit on, so the lag comes off and `quantize`
    /// works. What a fixed grid never gets is the whole-bar shift a tracked
    /// grid uses to keep ticks positive after that correction: a note pulled
    /// before zero is clamped to zero instead.
    public static func fixed(bpm: Double, beatsPerBar: Int? = 4, duration: Double) -> BeatGrid {
        let beat = 60 / bpm
        let count = max(2, Int((duration / beat).rounded(.up)) + 1)
        return BeatGrid(bpm: bpm, beatsPerBar: beatsPerBar, firstDownbeat: 0,
                        beats: (0..<count).map { Double($0) * beat }, isFixed: true)
    }

    /// What is written when nothing was detected: 120 BPM, no meter.
    public static let placeholder = BeatGrid(bpm: 120, beatsPerBar: nil, firstDownbeat: 0)

    /// This grid with its tempo snapped into `range` by a musical ratio.
    ///
    /// The beat list is rebuilt at the new tempo from the first downbeat across
    /// the tracked span, so the onset-lag measurement and `quantize` see the
    /// corrected pulse; beats per bar and the downbeat stay as tracked, which
    /// halves or doubles the bar in seconds rather than turning 4/4 into 8/4.
    /// A tempo already inside the range, a fixed grid, or one no ratio can
    /// bring inside comes back unchanged.
    ///
    /// When several ratios land inside — 89 reaches 120…190 as 133.5 (×3/2) and
    /// as 178 (×2) — `onsets` decide: at the true tempo the notes, and above all
    /// the drum hits, sit on the beats; at a wrong ratio they smear around the
    /// circle. The candidate whose beat-level concentration is highest wins.
    /// Without onsets, or with fewer than twenty, the smallest ratio wins.
    public func snapped(into range: ClosedRange<Double>, onsets: [Double]? = nil) -> BeatGrid {
        guard !isFixed else { return self }
        let candidates = BeatGridMath.snapCandidates(bpm: bpm, into: range)
        guard let first = candidates.first else { return self }
        var choice = first
        if candidates.count > 1, let onsets, onsets.count >= 20 {
            var best = -1.0
            for factor in candidates {
                let candidate = rebuilt(at: bpm * factor)
                guard let beats = candidate.beats else { continue }
                let phases = BeatGridMath.onsetPhases(onsets: onsets, beats: beats)
                guard phases.count >= 20 else { continue }
                let score = BeatGridMath.phaseConcentration(phases, subdivision: 1).concentration
                if score > best + 1e-9 { best = score; choice = factor }
            }
        }
        return rebuilt(at: bpm * choice)
    }

    /// The same span and downbeat at another tempo.
    func rebuilt(at newBPM: Double) -> BeatGrid {
        let step = 60 / newBPM
        var list: [Double]? = nil
        if let beats, let first = beats.first, let last = beats.last {
            var built: [Double] = []
            var t = firstDownbeat
            while t - step >= first - 1e-9 { t -= step }
            while t <= last + 1e-9 { built.append(t); t += step }
            list = built
        }
        return BeatGrid(bpm: newBPM, beatsPerBar: beatsPerBar, firstDownbeat: firstDownbeat, beats: list)
    }

    /// Seconds per bar, when the meter is known.
    public var barSeconds: Double? { beatsPerBar.map { Double($0) * 60 / bpm } }

    /// Seconds to delay every note so bar lines land on downbeats.
    ///
    /// MIDI has no pickup measure — bar 1 starts at tick 0 — so the only way to
    /// put a bar line on the first downbeat is to shift the music later. Whole
    /// bars (or beats, without a meter) are added until the shift reaches
    /// `minimumShift`, so a caller that moved notes earlier by the onset delay
    /// still gets non-negative ticks.
    public func barOffset(minimumShift: Double = 0) -> Double {
        var step: Double
        var offset: Double
        if let bar = barSeconds {
            step = bar
            offset = (bar - firstDownbeat.truncatingRemainder(dividingBy: bar)).truncatingRemainder(dividingBy: bar)
            if offset < 0 { offset += bar }
        } else {
            step = 60 / bpm
            offset = 0
        }
        if offset < minimumShift {
            offset += step * ((minimumShift - offset) / step).rounded(.up)
        }
        return offset
    }

    /// This grid with the onsets' lag against it measured and filled in.
    public func withOnsetDelay(onsets: [Double]) -> BeatGrid {
        var grid = self
        if let beats, let measured = BeatGridMath.estimateOnsetDelay(onsets: onsets, beats: beats, bpm: bpm) {
            grid.onsetDelay = measured.seconds
            grid.beatSubdivision = measured.subdivision
        } else {
            grid.onsetDelay = 0
            grid.beatSubdivision = nil
        }
        return grid
    }
}
