//
//  FilenameTempoTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//

import MIDIFileKit
import XCTest
@testable import MusicTranscriber

final class FilenameTempoTests: XCTestCase {

    func testReadsPackNames() {
        XCTAssertEqual(FilenameTempo.bpm(inFilename: "VENDOR_PK2_123_bass_loop_bill_C.wav"), 123)
        XCTAssertEqual(FilenameTempo.bpm(inFilename: "Drums 128 Cmin.wav"), 128)
        XCTAssertEqual(FilenameTempo.bpm(inFilename: "bass_140bpm.wav"), 140)
        XCTAssertEqual(FilenameTempo.bpm(inFilename: "Pad 174 BPM - 24bit.wav"), 174)
        XCTAssertEqual(FilenameTempo.bpm(inFilename: "loop-92.5bpm.aif"), 92.5)
    }

    func testIgnoresImplausibleNumbers() {
        XCTAssertNil(FilenameTempo.bpm(inFilename: "kick_24bit_2024.wav"))
        XCTAssertNil(FilenameTempo.bpm(inFilename: "PK2_pad.wav"))
        XCTAssertNil(FilenameTempo.bpm(inFilename: "song.mp3"))
    }

    func testFixedGridIsNeverShifted() {
        let grid = BeatGrid.fixed(bpm: 123, duration: 7.8)
        XCTAssertEqual(grid.beats?.count, 17, "a beat every 0.488 s across 7.8 s, both ends included")
        XCTAssertEqual(grid.barOffset(), 0)
        // Onsets 20 ms late on sixteenths, inside the loop: the subdivision and
        // the lag are measured, and the file is not bar-shifted to absorb it.
        let onsets = (0..<60).map { Double($0) * 60 / 123 / 4 + 0.02 }
        let measured = grid.withOnsetDelay(onsets: onsets)
        XCTAssertEqual(measured.beatSubdivision, 4)
        XCTAssertEqual(measured.onsetDelay ?? 0, 0.02, accuracy: 1e-4)
        let notes = onsets.map { TranscribedNote(pitch: 60, onset: $0, offset: $0 + 0.1, instrument: "acoustic_piano", program: 0, isDrum: false) }
        let file = MIDIAssembly.file(notes: notes, grid: measured)
        XCTAssertEqual(file.notes.first?.startTicks, 0, "the lag comes off and the first note is clamped to the start, not pushed a bar later")
        XCTAssertEqual(file.notes[1].startTicks, 120, "a sixteenth at 480 tpqn, on the grid")
    }
}
