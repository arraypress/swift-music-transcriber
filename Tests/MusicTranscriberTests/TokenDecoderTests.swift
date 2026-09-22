//
//  TokenDecoderTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  The decoder against upstream's decode_model_tokens on a real token stream:
//  three chunks of the medium model on the demo clip, prompts included.
//

import XCTest
@testable import MusicTranscriber

final class TokenDecoderTests: XCTestCase {

    struct Chunk: Decodable { let prompt: [Int]; let tokens: [Int] }
    struct Event: Decodable {
        let type: String
        let pitch: Int?; let start_time: Double?; let index: Int?; let instrument: String?
        let end_time: Double?; let start_event_index: Int?
    }
    struct File: Decodable { let chunks: [Chunk]; let events: [Event] }

    func testMatchesUpstreamEventStream() throws {
        let file = try Fixture.json("decoder.json", as: File.self)
        var decoder = TokenDecoder()
        var got: [TranscriptionEvent] = []
        for (i, chunk) in file.chunks.enumerated() {
            got += decoder.beginChunk(seekTime: Double(i) * 5, nextSeekTime: i + 1 < file.chunks.count ? Double(i + 1) * 5 : nil)
            for token in chunk.prompt + chunk.tokens { got += decoder.feed(token) }
        }
        got += decoder.finish()

        XCTAssertEqual(got.count, file.events.count)
        for (g, e) in zip(got, file.events) {
            switch g {
            case .noteStart(let s):
                XCTAssertEqual(e.type, "start")
                XCTAssertEqual(s.pitch, e.pitch); XCTAssertEqual(s.index, e.index); XCTAssertEqual(s.instrument, e.instrument)
                XCTAssertEqual(s.startTime, e.start_time!, accuracy: 1e-9)
            case .noteEnd(let n):
                XCTAssertEqual(e.type, "end")
                XCTAssertEqual(n.startEventIndex, e.start_event_index)
                XCTAssertEqual(n.endTime, e.end_time!, accuracy: 1e-9)
            case .progress:
                XCTFail("the decoder never emits progress")
            }
        }
    }

    /// The prompt for chunk N+1 is what chunk N left open — upstream's forcing invariant.
    func testOpenKeysFeedThePrologue() throws {
        let file = try Fixture.json("decoder.json", as: File.self)
        var decoder = TokenDecoder()
        _ = decoder.beginChunk(seekTime: 0, nextSeekTime: 5)
        for token in file.chunks[0].tokens { _ = decoder.feed(token) }
        _ = decoder.beginChunk(seekTime: 5, nextSeekTime: 10)
        XCTAssertEqual(EventVocabulary.tiePrologue(for: decoder.openKeys), file.chunks[1].prompt)
    }

    func testMalformedChunkClosesEverything() {
        var decoder = TokenDecoder()
        _ = decoder.beginChunk(seekTime: 0, nextSeekTime: 5)
        var events = decoder.feed(EventVocabulary.id(.tie))
        events += decoder.feed(EventVocabulary.id(.program(0)))
        events += decoder.feed(EventVocabulary.id(.velocity(1)))
        events += decoder.feed(EventVocabulary.id(.pitch(60)))
        XCTAssertEqual(events.count, 1)
        // Next chunk opens with a shift before any tie: close at the boundary, drop the rest.
        events = decoder.beginChunk(seekTime: 5, nextSeekTime: nil)
        events += decoder.feed(EventVocabulary.id(.shift(10)))
        events += decoder.feed(EventVocabulary.id(.velocity(1)))
        events += decoder.feed(EventVocabulary.id(.pitch(62)))
        XCTAssertEqual(events.count, 1)
        if case .noteEnd(let end) = events[0] { XCTAssertEqual(end.endTime, 5) } else { XCTFail() }
        XCTAssertTrue(decoder.openKeys.isEmpty)
    }
}
