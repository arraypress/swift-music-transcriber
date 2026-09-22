//
//  MIDIAssembly.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Notes and a grid to a Standard MIDI File, laid out as upstream lays it out.
//

import Foundation
import MIDIFileKit

/// Builds the MIDI file.
///
/// The layout is upstream's: one track per instrument, named after it, on
/// channels 1–9 and 11–16 in order of first appearance, drums on channel 10;
/// every note at velocity 100 because the model has none; tempo from the grid;
/// a time signature only when the meter was agreed; and a marker recording how
/// far the music was shifted to put bar 1 on a downbeat.
public enum MIDIAssembly {

    /// Velocity written for every note.
    public static let velocity = 100

    /// Ticks per quarter note.
    public static let ticksPerQuarterNote = 480

    /// Build the file.
    ///
    /// - Parameters:
    ///   - notes: cleaned notes, in seconds.
    ///   - grid: the detected grid, or nil for the 120 BPM placeholder.
    ///   - quantize: snap onsets and offsets to the grid's subdivision, when it
    ///     has one. What sheet music has to be engraved from and not what
    ///     anyone wants to listen to.
    public static func file(notes input: [TranscribedNote], grid: BeatGrid?, quantize: Bool = false) -> MIDIFile {
        var grid = grid ?? .placeholder
        if grid.onsetDelay == nil {
            grid = grid.withOnsetDelay(onsets: input.map(\.onset))
        }
        let delay = grid.onsetDelay ?? 0
        let offset = grid.barOffset(minimumShift: delay)
        var notes = BeatGridMath.shifted(input, by: -delay)
        if quantize, let subdivision = grid.beatSubdivision {
            notes = BeatGridMath.quantized(notes, step: 60 / grid.bpm / Double(subdivision), offset: offset)
        }

        // Ticks the way upstream computes them: from the tempo as MIDI stores it
        // (whole microseconds per beat), rounding on and off separately and
        // half-to-even, so a duration is the difference of two rounded ticks and
        // not a rounded difference. Off by one tick otherwise, on about a note in ten.
        let tempoMicros = (60_000_000 / grid.bpm).rounded(.toNearestOrEven)
        func ticks(_ seconds: Double) -> Int {
            Int((seconds * 1_000_000 / tempoMicros * Double(ticksPerQuarterNote)).rounded(.toNearestOrEven))
        }
        let tickBeat = 1 / Double(ticksPerQuarterNote)
        var composition = Composition(bpm: 60_000_000 / tempoMicros,
                                      timeSignature: (grid.beatsPerBar ?? 4, 4),
                                      ticksPerQuarterNote: ticksPerQuarterNote)
        if offset != 0 {
            composition.markers = [(0, BeatGridMath.barOffsetMarker + String(format: "%.4f", offset))]
        }

        // Tracks in order of first appearance in the time-sorted event stream:
        // (time, isDrum, program), which is how upstream's writer meets them.
        let ordered = notes.sorted { ($0.onset, $0.isDrum ? 1 : 0, $0.program) < ($1.onset, $1.isDrum ? 1 : 0, $1.program) }
        var programs: [Int] = []
        for n in ordered where !programs.contains(n.program) { programs.append(n.program) }
        var channels = Array(0...8) + Array(10...15)
        for program in programs {
            let mine = notes.filter { $0.program == program }
            let isDrum = program == InstrumentGroup.drumProgram
            let channel = isDrum ? 9 : (channels.isEmpty ? 15 : channels.removeFirst())
            let name = isDrum ? "drums"
                : (InstrumentGroup.forRepresentativeProgram(program)?.displayName ?? "program \(program)")
            composition.addTrack(name: name, channel: channel, program: isDrum ? 0 : program) { track in
                for n in mine {
                    let start = ticks(n.onset + offset)
                    let end = ticks(n.offset + offset)
                    track.note(n.pitch, atBeat: Double(start) * tickBeat,
                               lasting: Double(end - start) * tickBeat, velocity: velocity)
                }
            }
        }
        return composition.build()
    }

    /// The bytes of ``file(notes:grid:quantize:)``.
    public static func data(notes: [TranscribedNote], grid: BeatGrid?, quantize: Bool = false) throws -> Data {
        try MIDIWriter.data(for: file(notes: notes, grid: grid, quantize: quantize))
    }
}
