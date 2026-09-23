//
//  TempoSnapTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  The tracker's mishears are clean ratios; a declared range undoes them.
//

import XCTest
@testable import MusicTranscriber

final class TempoSnapTests: XCTestCase {

    func testRatiosMeasuredInTheWild() {
        XCTAssertEqual(BeatGridMath.snapCandidates(bpm: 89, into: 120...190), [1.5, 2], "the Makina track: two ratios land, the notes must decide")
        XCTAssertEqual(BeatGridMath.snapMultiplier(bpm: 77.5, into: 120...190), 2, "the demo clip")
        XCTAssertEqual(BeatGridMath.snapMultiplier(bpm: 82, into: 120...135), 1.5, "the 123 BPM bassline heard at 2/3")
        XCTAssertEqual(BeatGridMath.snapMultiplier(bpm: 128, into: 120...135), 1, "already inside")
        XCTAssertEqual(BeatGridMath.snapMultiplier(bpm: 50, into: 120...135), 1, "no ratio lands inside: left alone")
    }

    /// Kicks every 178th of a minute, tracked at 89: with 120…190 both 133.5 and
    /// 178 are reachable, and the kicks pick 178 because they sit on its beats.
    func testKicksChooseBetweenCandidates() {
        let tracked = (0..<40).map { 0.3 + Double($0) * 60 / 89 }
        let grid = BeatGrid(bpm: 89, beatsPerBar: 4, firstDownbeat: 0.3, beats: tracked)
        XCTAssertEqual(grid.snapped(into: 120...190).bpm, 133.5, accuracy: 1e-9, "without evidence the smaller correction wins")
        let kicks = (0..<70).map { 0.3 + Double($0) * 60 / 178 + 0.004 }
        let snapped = grid.snapped(into: 120...190, onsets: kicks)
        XCTAssertEqual(snapped.bpm, 178, accuracy: 1e-9)
        XCTAssertEqual(snapped.beatsPerBar, 4)
        XCTAssertEqual(snapped.firstDownbeat, 0.3)
        XCTAssertEqual(snapped.beats?.count, 79, "twice the pulse across the same span")
        XCTAssertEqual(snapped.beats?[1] ?? 0, 0.3 + 60.0 / 178, accuracy: 1e-9)
        XCTAssertFalse(snapped.isFixed)
    }

    func testFixedAndInRangeGridsAreUntouched() {
        let fixed = BeatGrid.fixed(bpm: 82, duration: 8)
        XCTAssertEqual(fixed.snapped(into: 120...135).bpm, 82)
        let grid = BeatGrid(bpm: 128, beatsPerBar: 4, firstDownbeat: 0)
        XCTAssertEqual(grid.snapped(into: 120...135), grid)
    }
}
