//
//  VocabularyTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//

import XCTest
@testable import MusicTranscriber

final class VocabularyTests: XCTestCase {

    func testLayoutMatchesUpstream() {
        XCTAssertEqual(EventVocabulary.count, 1393)
        XCTAssertEqual(EventVocabulary.event(0), .pad)
        XCTAssertEqual(EventVocabulary.event(1), .endOfChunk)
        XCTAssertEqual(EventVocabulary.event(3), .shift(0))
        XCTAssertEqual(EventVocabulary.event(1003), .shift(1000))
        XCTAssertEqual(EventVocabulary.event(1004), .pitch(0))
        XCTAssertEqual(EventVocabulary.event(1131), .pitch(127))
        XCTAssertEqual(EventVocabulary.event(1132), .velocity(0))
        XCTAssertEqual(EventVocabulary.event(1133), .velocity(1))
        XCTAssertEqual(EventVocabulary.event(1134), .tie)
        XCTAssertEqual(EventVocabulary.event(1135), .program(0))
        XCTAssertEqual(EventVocabulary.event(1264), .program(129))
        XCTAssertEqual(EventVocabulary.event(1265), .drum(0))
        XCTAssertEqual(EventVocabulary.event(1392), .drum(127))
        XCTAssertEqual(EventVocabulary.event(1393), .unknown)
    }

    func testRoundTrip() {
        for id in 0..<EventVocabulary.count {
            XCTAssertEqual(EventVocabulary.id(EventVocabulary.event(id)), id)
        }
    }

    /// The tie prologue lists programs once each, pitches sorted, then `tie`.
    func testTiePrologue() {
        let tokens = EventVocabulary.tiePrologue(for: [(program: 33, pitch: 40), (program: 0, pitch: 60), (program: 0, pitch: 64)])
        XCTAssertEqual(tokens.map(EventVocabulary.event), [.program(0), .pitch(60), .pitch(64), .program(33), .pitch(40), .tie])
        XCTAssertEqual(EventVocabulary.tiePrologue(for: []), [1134])
    }

    func testForbiddenTokens() {
        let piano = InstrumentGroup.named("acoustic_piano")!
        let forbidden = Set(EventVocabulary.forbiddenTokens(allowing: [piano]))
        XCTAssertFalse(forbidden.contains(EventVocabulary.id(.program(0))), "the allowed group's representative stays")
        XCTAssertTrue(forbidden.contains(EventVocabulary.id(.program(1))), "other programs go, even inside the group")
        XCTAssertTrue(forbidden.contains(EventVocabulary.id(.drum(36))), "drums go unless listed")
        XCTAssertEqual(forbidden.count, 129 + 128)
        let withDrums = Set(EventVocabulary.forbiddenTokens(allowing: [piano, InstrumentGroup.named("drums")!]))
        XCTAssertFalse(withDrums.contains(EventVocabulary.id(.drum(36))))
    }
}
