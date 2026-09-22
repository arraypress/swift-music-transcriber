//
//  NoteCleanup.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  From an event stream to a clean note list, the way upstream's writer does it.
//

import Foundation

/// Pure functions between events and notes.
public enum NoteCleanup {

    /// Pair every start with its end.
    ///
    /// Progress events are ignored. Drum notes keep upstream's pseudo-program 128.
    public static func notes(from events: [TranscriptionEvent]) -> [TranscribedNote] {
        var open: [Int: TranscriptionEvent.NoteStart] = [:]
        var notes: [TranscribedNote] = []
        for event in events {
            switch event {
            case .noteStart(let start):
                open[start.index] = start
            case .noteEnd(let end):
                guard let start = open.removeValue(forKey: end.startEventIndex) else { continue }
                notes.append(TranscribedNote(pitch: start.pitch, onset: start.startTime, offset: end.endTime,
                                             instrument: start.instrument,
                                             program: start.isDrum ? InstrumentGroup.drumProgram : start.program,
                                             isDrum: start.isDrum))
            case .progress:
                continue
            }
        }
        return notes
    }

    /// Upstream's `validate_notes(fix=True)` then `trim_overlapping_notes(sort=True)`.
    ///
    /// A note whose end precedes its start gets the 10 ms floor; a pitched note
    /// shorter than 10 ms is stretched to it; where two notes on the same
    /// (program, pitch) overlap, the earlier one is cut at the later one's onset
    /// and anything left with no length is dropped. Then sorted by
    /// (onset, isDrum, program, pitch, offset).
    public static func cleaned(_ input: [TranscribedNote]) -> [TranscribedNote] {
        var notes = input.map { note -> TranscribedNote in
            var n = note
            if n.onset > n.offset {
                n.offset = max(n.offset, n.onset + TranscribedNote.minimumDuration)
            } else if !n.isDrum && n.offset - n.onset < TranscribedNote.minimumDuration {
                n.offset = n.onset + TranscribedNote.minimumDuration
            }
            return n
        }
        guard notes.count > 1 else { return notes }

        var byChannel: [Channel: [Int]] = [:]
        for (i, n) in notes.enumerated() {
            byChannel[Channel(program: n.program, pitch: n.pitch, isDrum: n.isDrum), default: []].append(i)
        }
        var keep: [TranscribedNote] = []
        for (_, indices) in byChannel {
            let sorted = indices.sorted { notes[$0].onset < notes[$1].onset }
            for k in 1..<max(1, sorted.count) where notes[sorted[k - 1]].offset > notes[sorted[k]].onset {
                notes[sorted[k - 1]].offset = notes[sorted[k]].onset
            }
            keep.append(contentsOf: sorted.map { notes[$0] }.filter { $0.onset < $0.offset })
        }
        return sorted(keep)
    }

    /// Upstream's `sort_notes` order.
    public static func sorted(_ notes: [TranscribedNote]) -> [TranscribedNote] {
        notes.sorted {
            ($0.onset, $0.isDrum ? 1 : 0, $0.program, $0.pitch, $0.offset)
                < ($1.onset, $1.isDrum ? 1 : 0, $1.program, $1.pitch, $1.offset)
        }
    }

    private struct Channel: Hashable { let program: Int; let pitch: Int; let isDrum: Bool }
}
