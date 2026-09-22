//
//  BeatGridMathTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//

import XCTest
@testable import MusicTranscriber

final class BeatGridMathTests: XCTestCase {

    func testFitTempo() {
        let beats = (0..<32).map { 0.25 + Double($0) * 0.5 }   // 120 BPM, offset a quarter second
        let (bpm, residual) = BeatGridMath.fitTempo(beats)
        XCTAssertEqual(bpm, 120, accuracy: 1e-9)
        XCTAssertEqual(residual, 0, accuracy: 1e-9)
    }

    func testMeter() {
        let beats = (0..<32).map { Double($0) * 0.5 }
        let bars = stride(from: 0.0, to: 16, by: 2).map { $0 }        // every 4 beats
        XCTAssertEqual(BeatGridMath.inferBeatsPerBar(beats: beats, downbeats: bars), 4)
        let waltz = stride(from: 0.0, to: 16, by: 1.5).map { $0 }     // every 3 beats
        XCTAssertEqual(BeatGridMath.inferBeatsPerBar(beats: beats, downbeats: waltz), 3)
        XCTAssertNil(BeatGridMath.inferBeatsPerBar(beats: beats, downbeats: [0, 2, 3.5, 6, 7, 9.5]), "bars that disagree write no meter")
    }

    func testNoSteadyTempoThrows() {
        let wobbly = (0..<16).map { Double($0) * 0.5 + ($0 % 2 == 0 ? 0.1 : -0.1) }
        XCTAssertThrowsError(try BeatGridMath.grid(beats: wobbly, downbeats: []))
        XCTAssertThrowsError(try BeatGridMath.grid(beats: [0, 0.5, 1], downbeats: []))
    }

    /// Onsets 20 ms late on every eighth note are measured as such.
    func testOnsetDelay() {
        let beats = (0..<64).map { Double($0) * 0.5 }
        let onsets = (0..<120).map { Double($0) * 0.25 + 0.02 }
        let delay = BeatGridMath.estimateOnsetDelay(onsets: onsets, beats: beats, bpm: 120)
        XCTAssertEqual(delay?.subdivision, 2)
        XCTAssertEqual(delay?.seconds ?? 0, 0.02, accuracy: 1e-6)
        XCTAssertGreaterThan(delay?.concentration ?? 0, 0.99)
    }

    func testBarOffset() {
        let grid = BeatGrid(bpm: 120, beatsPerBar: 4, firstDownbeat: 0.5)
        XCTAssertEqual(grid.barOffset(), 1.5, accuracy: 1e-9, "shift forward until a bar line lands on 0.5 s")
        XCTAssertEqual(grid.barOffset(minimumShift: 1.6), 3.5, accuracy: 1e-9, "whole bars added to reach the minimum")
        XCTAssertEqual(BeatGrid.placeholder.barOffset(), 0)
    }

    func testQuantize() {
        let notes = [TranscribedNote(pitch: 60, onset: 0.26, offset: 0.27, instrument: "acoustic_piano", program: 0, isDrum: false)]
        let q = BeatGridMath.quantized(notes, step: 0.25, offset: 0)
        XCTAssertEqual(q[0].onset, 0.25, accuracy: 1e-9)
        XCTAssertEqual(q[0].offset, 0.5, accuracy: 1e-9, "a note shorter than half a step gets one step")
    }
}
