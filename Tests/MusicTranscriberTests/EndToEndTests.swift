//
//  EndToEndTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  The whole pipeline against upstream's medium fp32 run on the demo clip.
//  Skipped unless SCRIBE_MODEL names a medium asset: the weights are gated and
//  1.2 GB, so they are never in the repo.
//

import XCTest
@testable import MusicTranscriber

final class EndToEndTests: XCTestCase {

    /// Same 16 kHz samples in, same events out — token for token through the
    /// decoder, so the events must be identical, not merely close.
    func testMediumReproducesUpstreamEvents() async throws {
        guard let model = try? ModelLocator.resolve(variant: .medium, precision: .float32) else {
            throw XCTSkip("no medium model installed; set SCRIBE_MODEL to run the end-to-end check")
        }
        let file = try Fixture.json("decoder.json", as: TokenDecoderTests.File.self)
        let transcriber = try await MusicTranscriber(model: model)
        guard transcriber.decoder.variant == .medium else {
            throw XCTSkip("the installed model is not medium; the fixture is a medium run")
        }
        let result = try await transcriber.transcribe(try Fixture.url("demo_16k.wav"), tempo: .off)

        XCTAssertEqual(result.events.count, file.events.count)
        for (record, expected) in zip(result.events, file.events) {
            switch record {
            case .start(let s):
                XCTAssertEqual(expected.type, "start")
                XCTAssertEqual(s.pitch, expected.pitch); XCTAssertEqual(s.instrument, expected.instrument)
                XCTAssertEqual(s.startTime, expected.start_time ?? -1, accuracy: 1e-9)
            case .end(let e):
                XCTAssertEqual(expected.type, "end")
                XCTAssertEqual(e.startEventIndex, expected.start_event_index)
                XCTAssertEqual(e.endTime, expected.end_time ?? -1, accuracy: 1e-9)
            }
        }
        XCTAssertEqual(result.notes.count, 173)
        XCTAssertTrue(result.warnings.isEmpty, "\(result.warnings)")
    }
}
