//
//  EventVocabulary.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The token table: what each of the model's 1,393 output ids means.
//

import Foundation

/// The MT3-style event vocabulary, laid out exactly as upstream's `build_event_vocab`.
///
/// The order is fixed and load-bearing: special tokens, then 1,001 time shifts,
/// then pitch, velocity, tie, program and drum ranges. Token `i` means
/// ``event(_:)`` and nothing else; the model was trained on these ids.
public enum EventVocabulary {

    /// One decoded token.
    public enum Event: Hashable, Sendable {
        case pad, endOfChunk, unknown
        /// Move the clock to `steps` frames past the chunk start (10 ms frames).
        case shift(Int)
        /// A note-on or note-off for this pitch, depending on the current velocity state.
        case pitch(Int)
        /// 1 = following pitches are note-ons, 0 = note-offs.
        case velocity(Int)
        /// End of the tie prologue.
        case tie
        /// Following pitches belong to this General MIDI program.
        case program(Int)
        /// An instantaneous drum hit on this note number.
        case drum(Int)
    }

    /// Frames per second of the shift clock.
    public static let frameRate = 100

    /// The largest shift, in frames: a 5-second chunk is 500, plus headroom.
    public static let maxShiftSteps = 1001

    // Range starts, in id order. Values inside a range are contiguous.
    static let shiftBase = 3
    static let pitchBase = shiftBase + maxShiftSteps          // 1004
    static let velocityBase = pitchBase + 128                  // 1132
    static let tieID = velocityBase + 2                        // 1134
    static let programBase = tieID + 1                         // 1135
    static let drumBase = programBase + 130                    // 1265

    /// Number of real tokens. Checkpoints may be wider (``ModelVariant/card``);
    /// anything at or past this index is masked before sampling.
    public static let count = drumBase + 128                   // 1393

    /// The id of the end-of-chunk token.
    public static let endOfChunk = 1

    /// Decode an id.
    public static func event(_ id: Int) -> Event {
        switch id {
        case 0: return .pad
        case 1: return .endOfChunk
        case 2: return .unknown
        case shiftBase..<pitchBase: return .shift(id - shiftBase)
        case pitchBase..<velocityBase: return .pitch(id - pitchBase)
        case velocityBase..<tieID: return .velocity(id - velocityBase)
        case tieID: return .tie
        case programBase..<drumBase: return .program(id - programBase)
        case drumBase..<count: return .drum(id - drumBase)
        default: return .unknown
        }
    }

    /// Encode an event.
    public static func id(_ event: Event) -> Int {
        switch event {
        case .pad: return 0
        case .endOfChunk: return 1
        case .unknown: return 2
        case .shift(let n): return shiftBase + n
        case .pitch(let p): return pitchBase + p
        case .velocity(let v): return velocityBase + v
        case .tie: return tieID
        case .program(let p): return programBase + p
        case .drum(let d): return drumBase + d
        }
    }

    /// The tie prologue that declares `openKeys` as sustained into a new chunk.
    ///
    /// The layout matches the training encoder: pairs sorted by (program, pitch),
    /// each program token once for its run of pitches, then `tie`. Teacher-
    /// forcing this at the start of a chunk is what "prelude forcing" means.
    public static func tiePrologue(for openKeys: [(program: Int, pitch: Int)]) -> [Int] {
        var tokens: [Int] = []
        var current: Int?
        for key in openKeys.sorted(by: { ($0.program, $0.pitch) < ($1.program, $1.pitch) }) {
            if key.program != current {
                tokens.append(id(.program(key.program)))
                current = key.program
            }
            tokens.append(id(.pitch(key.pitch)))
        }
        tokens.append(tieID)
        return tokens
    }

    /// Ids that may never be sampled when only `groups` are allowed.
    ///
    /// Every program token is forbidden unless it is an allowed group's
    /// representative program; drum tokens are forbidden unless drums are listed.
    /// Timing, pitch, velocity, tie and special tokens are never forbidden.
    public static func forbiddenTokens(allowing groups: [InstrumentGroup]) -> [Int] {
        let allowedPrograms = Set(groups.filter { !$0.isDrums }.map(\.representativeProgram))
        let allowDrums = groups.contains { $0.isDrums }
        var forbidden: [Int] = []
        for program in 0..<130 where !allowedPrograms.contains(program) {
            forbidden.append(programBase + program)
        }
        if !allowDrums {
            forbidden.append(contentsOf: drumBase..<(drumBase + 128))
        }
        return forbidden
    }
}
