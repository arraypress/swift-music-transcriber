//
//  PianoPostprocessing.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Upstream's RegressionPostProcessor and piano_vad, line for line: the
//  segment stitching, the regression-to-event step (a local maximum above the
//  threshold with monotonic neighbours becomes an onset or offset, and the
//  values either side of the peak give a sub-frame shift), the note state
//  machine (onset, then the earlier of "frame went quiet" and "offset
//  predicted", with a 6-second cap) and the pedal one. Python's `if bgn:`
//  treats frame 0 as no onset; that is kept, because the events must match.
//

import Foundation

/// The seven framewise outputs of the piano model, after stitching, and the
/// events upstream derives from them.
public enum PianoPostprocessing {

    public static let classes = 88
    /// MIDI note of the lowest piano key, A0.
    public static let beginNote = 21
    public static let framesPerSecond = 100
    public static let onsetThreshold: Float = 0.3
    public static let offsetThreshold: Float = 0.3
    public static let frameThreshold: Float = 0.1
    public static let pedalOffsetThreshold: Float = 0.2
    public static let pedalFrameThreshold: Float = 0.5
    static let velocityScale: Float = 128

    /// Framewise outputs over a whole recording, `[frames][classes]` for notes and `[frames]` for the pedal.
    public struct Outputs: Sendable {
        public var frames: Int
        public var regOnset: [Float]
        public var regOffset: [Float]
        public var frame: [Float]
        public var velocity: [Float]
        public var pedalOnset: [Float]
        public var pedalOffset: [Float]
        public var pedalFrame: [Float]
    }

    public struct NoteEvent: Equatable, Sendable {
        public var onset: Float
        public var offset: Float
        public var midiNote: Int
        public var velocity: Int
    }

    public struct PedalEvent: Equatable, Sendable {
        public var onset: Float
        public var offset: Float
    }

    // MARK: - Stitching

    /// Upstream's `deframe`: one segment is returned whole; several drop their
    /// last frame, then the first contributes its first three quarters, the
    /// middle ones their middle half, the last its final three quarters.
    public static func deframe(_ segments: [[Float]], segmentFrames: Int, width: Int) -> (values: [Float], frames: Int) {
        if segments.count == 1 { return (segments[0], segmentFrames) }
        let kept = segmentFrames - 1
        precondition(kept % 4 == 0)
        let quarter = kept / 4
        var out: [Float] = []
        out.append(contentsOf: segments[0][0..<(3 * quarter * width)])
        for i in 1..<(segments.count - 1) {
            out.append(contentsOf: segments[i][(quarter * width)..<(3 * quarter * width)])
        }
        out.append(contentsOf: segments[segments.count - 1][(quarter * width)..<(kept * width)])
        return (out, out.count / width)
    }

    // MARK: - Regression to binary + shift

    /// `get_binarized_output_from_regression`: per class, a frame above the
    /// threshold whose `neighbour` frames on both sides fall away monotonically
    /// is an event; the shift is the paper's Section III-D estimate.
    static func binarize(_ x: [Float], frames: Int, width: Int, threshold: Float, neighbour: Int) -> (binary: [Bool], shift: [Float]) {
        var binary = [Bool](repeating: false, count: frames * width)
        var shift = [Float](repeating: 0, count: frames * width)
        guard frames > 2 * neighbour else { return (binary, shift) }
        for k in 0..<width {
            for n in neighbour..<(frames - neighbour) {
                let here = x[n * width + k]
                guard here > threshold else { continue }
                var monotonic = true
                for i in 0..<neighbour {
                    if x[(n - i) * width + k] < x[(n - i - 1) * width + k] { monotonic = false }
                    if x[(n + i) * width + k] < x[(n + i + 1) * width + k] { monotonic = false }
                }
                guard monotonic else { continue }
                binary[n * width + k] = true
                let before = x[(n - 1) * width + k], after = x[(n + 1) * width + k]
                shift[n * width + k] = before > after
                    ? (after - before) / (here - after) / 2
                    : (after - before) / (here - before) / 2
            }
        }
        return (binary, shift)
    }

    // MARK: - Events

    /// `output_dict_to_midi_events`: note and pedal events in seconds.
    public static func events(from o: Outputs) -> (notes: [NoteEvent], pedals: [PedalEvent]) {
        let (onset, onsetShift) = binarize(o.regOnset, frames: o.frames, width: classes, threshold: onsetThreshold, neighbour: 2)
        let (offset, offsetShift) = binarize(o.regOffset, frames: o.frames, width: classes, threshold: offsetThreshold, neighbour: 4)
        var notes: [NoteEvent] = []
        for k in 0..<classes {
            let column = { (a: [Float]) in (0..<o.frames).map { a[$0 * classes + k] } }
            let columnB = { (a: [Bool]) in (0..<o.frames).map { a[$0 * classes + k] } }
            for t in detectNotes(frame: column(o.frame), onset: columnB(onset), onsetShift: column(onsetShift),
                                 offset: columnB(offset), offsetShift: column(offsetShift), velocity: column(o.velocity)) {
                // numpy: (bgn + shift) / fps in float64, stored as float32; velocity truncated from float32 × 128.
                notes.append(NoteEvent(onset: Float((Double(t.begin) + Double(t.onsetShift)) / Double(framesPerSecond)),
                                       offset: Float((Double(t.end) + Double(t.offsetShift)) / Double(framesPerSecond)),
                                       midiNote: k + beginNote,
                                       velocity: Int(t.velocity * velocityScale)))
            }
        }
        let (pedalOff, pedalOffShift) = binarize(o.pedalOffset, frames: o.frames, width: 1, threshold: pedalOffsetThreshold, neighbour: 4)
        let pedals = detectPedals(frame: o.pedalFrame, offset: pedalOff, offsetShift: pedalOffShift).map {
            PedalEvent(onset: Float(Double($0.begin) / Double(framesPerSecond)),
                       offset: Float((Double($0.end) + Double($0.offsetShift)) / Double(framesPerSecond)))
        }
        return (notes, pedals)
    }

    struct Detected { var begin: Int; var end: Int; var onsetShift: Float; var offsetShift: Float; var velocity: Float }

    /// `note_detection_with_onset_offset_regress`, including Python's falsy zero.
    static func detectNotes(frame: [Float], onset: [Bool], onsetShift: [Float], offset: [Bool], offsetShift: [Float], velocity: [Float]) -> [Detected] {
        var out: [Detected] = []
        var bgn: Int? = nil, frameDisappear: Int? = nil, offsetOccur: Int? = nil
        func truthy(_ v: Int?) -> Bool { if let v { return v != 0 } else { return false } }
        let n = onset.count
        for i in 0..<n {
            if onset[i] {
                if truthy(bgn) {
                    // Consecutive onsets: the earlier note ends the frame before.
                    let fin = max(i - 1, 0)
                    out.append(Detected(begin: bgn!, end: fin, onsetShift: onsetShift[bgn!], offsetShift: 0, velocity: velocity[bgn!]))
                    frameDisappear = nil; offsetOccur = nil
                }
                bgn = i
            }
            if truthy(bgn), i > bgn! {
                if frame[i] <= frameThreshold, !truthy(frameDisappear) { frameDisappear = i }
                if offset[i], !truthy(offsetOccur) { offsetOccur = i }
                if truthy(frameDisappear) {
                    let fin: Int
                    if truthy(offsetOccur), offsetOccur! - bgn! > frameDisappear! - offsetOccur! {
                        fin = offsetOccur!
                    } else {
                        fin = frameDisappear!
                    }
                    out.append(Detected(begin: bgn!, end: fin, onsetShift: onsetShift[bgn!], offsetShift: offsetShift[fin], velocity: velocity[bgn!]))
                    bgn = nil; frameDisappear = nil; offsetOccur = nil
                }
                if truthy(bgn), i - bgn! >= 600 || i == n - 1 {
                    let fin = i
                    out.append(Detected(begin: bgn!, end: fin, onsetShift: onsetShift[bgn!], offsetShift: offsetShift[fin], velocity: velocity[bgn!]))
                    bgn = nil; frameDisappear = nil; offsetOccur = nil
                }
            }
        }
        return out.sorted { $0.begin < $1.begin }   // Python's sort is stable; so is this order of appends
    }

    struct DetectedPedal { var begin: Int; var end: Int; var offsetShift: Float }

    /// `pedal_detection_with_onset_offset_regress`: a rising frame value above
    /// 0.5 opens the pedal; a predicted offset, or ten frames after it went
    /// quiet, closes it.
    static func detectPedals(frame: [Float], offset: [Bool], offsetShift: [Float]) -> [DetectedPedal] {
        var out: [DetectedPedal] = []
        var bgn: Int? = nil, frameDisappear: Int? = nil, offsetOccur: Int? = nil
        func truthy(_ v: Int?) -> Bool { if let v { return v != 0 } else { return false } }
        guard frame.count > 1 else { return out }
        for i in 1..<frame.count {
            if frame[i] >= pedalFrameThreshold, frame[i] > frame[i - 1], !truthy(bgn) { bgn = i }
            if truthy(bgn), i > bgn! {
                if frame[i] <= pedalFrameThreshold, !truthy(frameDisappear) { frameDisappear = i }
                if offset[i], !truthy(offsetOccur) { offsetOccur = i }
                if truthy(offsetOccur) {
                    out.append(DetectedPedal(begin: bgn!, end: offsetOccur!, offsetShift: offsetShift[offsetOccur!]))
                    bgn = nil; frameDisappear = nil; offsetOccur = nil
                }
                if truthy(frameDisappear), i - frameDisappear! >= 10 {
                    out.append(DetectedPedal(begin: bgn!, end: frameDisappear!, offsetShift: offsetShift[frameDisappear!]))
                    bgn = nil; frameDisappear = nil; offsetOccur = nil
                }
            }
        }
        return out.sorted { $0.begin < $1.begin }
    }
}
