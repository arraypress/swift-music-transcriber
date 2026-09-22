//
//  TokenDecoder.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The chunk-decoding state machine: tokens in, note events out.
//
//  A port of upstream's OpenNoteTracker and decode_model_tokens, kept as one
//  object because two consumers must agree on it: the event stream, and the
//  prelude forcing that reads ``openKeys`` at every chunk boundary to build
//  the next prompt. One state machine serving both keeps them consistent by
//  construction — the same reason upstream did it that way.
//

import Foundation

/// Turns the model's token stream into ``TranscriptionEvent``s, chunk by chunk.
///
/// Usage per chunk: ``beginChunk(seekTime:nextSeekTime:)``, then ``feed(_:)``
/// for every token (prompt tokens included, end token excluded), and after the
/// last chunk ``finish()``. Every call returns the events it produced, in order.
public struct TokenDecoder {

    private let frameRate: Int

    // Open notes, (program, pitch) → the start event. Insertion-ordered so the
    // end-of-stream closes replay in onset order, as upstream's dict does.
    private var open: [(key: Key, start: TranscriptionEvent.NoteStart)] = []
    private var nextIndex = 0

    // Per-chunk state, reset at every boundary.
    private var seekTime = 0.0
    private var nextSeekTime: Double?
    private var startTick = 0
    private var tick = 0
    private var program: Int?
    private var velocity: Int?
    private var inPrologue = true
    private var skipRest = false
    private var tieSet: Set<Key> = []
    private var chunkStarted = false

    struct Key: Hashable { let program: Int; let pitch: Int }

    public init(frameRate: Int = EventVocabulary.frameRate) {
        self.frameRate = frameRate
    }

    /// The (program, pitch) pairs currently sounding, sorted — the next chunk's
    /// tie prologue when prelude forcing is on.
    public var openKeys: [(program: Int, pitch: Int)] {
        open.map { ($0.key.program, $0.key.pitch) }.sorted { ($0.0, $0.1) < ($1.0, $1.1) }
    }

    /// Start a chunk at `seekTime` seconds. `nextSeekTime` is the following
    /// chunk's start, nil for the last; events the model emits past it are dropped.
    ///
    /// If the previous chunk never closed its prologue (no `tie` token), its tie
    /// set is treated as empty and every open note ends at its boundary.
    public mutating func beginChunk(seekTime: Double, nextSeekTime: Double?) -> [TranscriptionEvent] {
        var events: [TranscriptionEvent] = []
        if chunkStarted && inPrologue {
            events = endAll(at: self.seekTime)
        }
        self.seekTime = seekTime
        self.nextSeekTime = nextSeekTime
        startTick = Int((seekTime * Double(frameRate)).rounded())
        tick = startTick
        program = nil
        velocity = nil
        inPrologue = true
        skipRest = false
        tieSet = []
        chunkStarted = true
        return events
    }

    /// Consume one token.
    public mutating func feed(_ token: Int) -> [TranscriptionEvent] {
        let event = EventVocabulary.event(token)

        if inPrologue {
            switch event {
            case .tie:
                // End of the tie section: close prior notes not sustained here.
                inPrologue = false
                velocity = nil
                let ended = open.filter { !tieSet.contains($0.key) }
                open.removeAll { !tieSet.contains($0.key) }
                return ended.map { end($0.start, at: seekTime) }
            case .shift:
                // No tie token: malformed chunk. Close everything, drop the rest.
                inPrologue = false
                skipRest = true
                return endAll(at: seekTime)
            case .program(let p):
                program = p
            case .pitch(let p):
                if let program { tieSet.insert(Key(program: program, pitch: p)) }
            default:
                break
            }
            return []
        }

        if skipRest { return [] }

        switch event {
        case .shift(let n):
            if n > 0 { tick = startTick + n }
        case .program(let p):
            program = p
        case .velocity(let v):
            velocity = v
        case .drum(let d):
            let time = Double(tick) / Double(frameRate)
            if let next = nextSeekTime, time >= next { return [] }
            let start = mint(pitch: d, time: time, program: InstrumentGroup.drumProgram, isDrum: true)
            return [.noteStart(start),
                    .noteEnd(.init(endTime: time + TranscribedNote.minimumDuration, startEventIndex: start.index))]
        case .pitch(let p):
            guard let program, let velocity else { return [] }
            let time = Double(tick) / Double(frameRate)
            if let next = nextSeekTime, time >= next { return [] }
            let key = Key(program: program, pitch: p)
            var events: [TranscriptionEvent] = []
            if let index = open.firstIndex(where: { $0.key == key }) {
                let entry = open.remove(at: index)
                events.append(end(entry.start, at: time))
            }
            if velocity > 0 {
                let start = mint(pitch: p, time: time, program: program, isDrum: false)
                open.append((key, start))
                events.append(.noteStart(start))
            }
            return events
        default:
            break
        }
        return []
    }

    /// End of stream: close anything still open.
    ///
    /// A well-formed final chunk gets the minimum-duration fallback; one that
    /// ended mid-prologue closes at its boundary, matching ``feed(_:)``.
    public mutating func finish() -> [TranscriptionEvent] {
        if chunkStarted && inPrologue { return endAll(at: seekTime) }
        let events = open.map { end($0.start, at: $0.start.startTime + TranscribedNote.minimumDuration) }
        open.removeAll()
        return events
    }

    // MARK: - Helpers

    private mutating func mint(pitch: Int, time: Double, program: Int, isDrum: Bool) -> TranscriptionEvent.NoteStart {
        defer { nextIndex += 1 }
        return .init(pitch: pitch, startTime: time, index: nextIndex,
                     instrument: isDrum ? "drums" : InstrumentGroup.label(forProgram: program),
                     program: program, isDrum: isDrum)
    }

    private func end(_ start: TranscriptionEvent.NoteStart, at time: Double) -> TranscriptionEvent {
        .noteEnd(.init(endTime: time, startEventIndex: start.index))
    }

    private mutating func endAll(at time: Double) -> [TranscriptionEvent] {
        let events = open.map { end($0.start, at: time) }
        open.removeAll()
        return events
    }
}
