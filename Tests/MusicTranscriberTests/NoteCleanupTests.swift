//
//  NoteCleanupTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//

import MIDIFileKit
import XCTest
@testable import MusicTranscriber

final class NoteCleanupTests: XCTestCase {

    struct Reference: Decodable { let pitch: Int; let onset: Double; let offset: Double; let program: Int; let is_drum: Bool }

    private func decodedNotes() throws -> [TranscribedNote] {
        let file = try Fixture.json("decoder.json", as: TokenDecoderTests.File.self)
        var decoder = TokenDecoder(leadingTies: .drop)   // upstream's reading, which these fixtures record
        var events: [TranscriptionEvent] = []
        for (i, chunk) in file.chunks.enumerated() {
            events += decoder.beginChunk(seekTime: Double(i) * 5, nextSeekTime: i + 1 < file.chunks.count ? Double(i + 1) * 5 : nil)
            for token in chunk.prompt + chunk.tokens { events += decoder.feed(token) }
        }
        events += decoder.finish()
        return NoteCleanup.cleaned(NoteCleanup.notes(from: events))
    }

    func testMatchesUpstreamNotes() throws {
        let reference = try Fixture.json("notes.json", as: [Reference].self)
        let notes = try decodedNotes()
        XCTAssertEqual(notes.count, reference.count)
        for (n, r) in zip(notes, reference) {
            XCTAssertEqual(n.pitch, r.pitch); XCTAssertEqual(n.program, r.program); XCTAssertEqual(n.isDrum, r.is_drum)
            XCTAssertEqual(n.onset, r.onset, accuracy: 1e-9); XCTAssertEqual(n.offset, r.offset, accuracy: 1e-9)
        }
    }

    /// The MIDI we write reads back with the same notes upstream's file has.
    func testMIDIMatchesUpstreamFile() throws {
        let ours = try MIDIReader.read(MIDIAssembly.data(notes: try decodedNotes(), grid: nil))
        let theirs = try MIDIReader.read(contentsOf: try Fixture.url("reference.mid"))
        XCTAssertEqual(ours.notes.count, theirs.notes.count)
        // Upstream repeats set_tempo on every track for MuseScore's sake, so `single` is nil there.
        XCTAssertEqual(ours.tempoMap.initial.bpm, theirs.tempoMap.initial.bpm, accuracy: 1e-6)
        XCTAssertEqual(ours.tracks.compactMap(\.name), theirs.tracks.compactMap(\.name))
        func key(_ n: Note) -> String { "\(n.startTicks)/\(n.pitch.number)/\(n.channel)" }
        let mine = Dictionary(ours.notes.map { (key($0), $0) }, uniquingKeysWith: { a, _ in a })
        for t in theirs.notes {
            guard let m = mine[key(t)] else { XCTFail("missing \(key(t))"); continue }
            XCTAssertEqual(m.durationTicks, t.durationTicks, "duration of \(key(t))")
            XCTAssertEqual(m.velocity, t.velocity)
        }
    }

    func testOverlapTrimAndFloor() {
        let a = TranscribedNote(pitch: 60, onset: 0, offset: 1, instrument: "acoustic_piano", program: 0, isDrum: false)
        let b = TranscribedNote(pitch: 60, onset: 0.5, offset: 0.503, instrument: "acoustic_piano", program: 0, isDrum: false)
        let cleaned = NoteCleanup.cleaned([a, b])
        XCTAssertEqual(cleaned.count, 2)
        XCTAssertEqual(cleaned[0].offset, 0.5, "the earlier note is cut at the later onset")
        XCTAssertEqual(cleaned[1].offset, 0.51, accuracy: 1e-9, "a 3 ms pitched note becomes 10 ms")
    }
}
