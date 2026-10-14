//
//  PipeCollectorTests.swift
//  MusicTranscriberTests
//
//  MuseScore's pipes are read while it runs: 1 MB on each stream,
//  past the 64 KB a pipe holds, comes back whole without waiting on a
//  timeout. No MuseScore — a shell writes the payload.
//

import Foundation
import XCTest
@testable import MusicTranscriber

final class PipeCollectorTests: XCTestCase {

    func testBothStreamsPastAPipeBufferComeBackWhole() throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", "head -c 1048576 /dev/zero | tr '\\0' a; head -c 1048576 /dev/zero | tr '\\0' b >&2"]
        let output = Pipe(), errors = Pipe()
        process.standardOutput = output
        process.standardError = errors
        let out = PipeCollector(output), err = PipeCollector(errors)
        try process.run()
        let started = Date()
        process.waitUntilExit()
        XCTAssertLessThan(Date().timeIntervalSince(started), 10)
        XCTAssertEqual(out.collected().count, 1_048_576)
        XCTAssertEqual(err.collected().count, 1_048_576)
    }

    func testTheMuseScoreRunnerSurvivesAFloodOfStderr() throws {
        // What MuseScore does: lots of log on stderr before anything on stdout.
        let started = Date()
        let result = try SheetMusic.run(URL(fileURLWithPath: "/bin/sh"),
                                        ["-c", "head -c 1048576 /dev/zero | tr '\\0' e >&2; echo done"])
        XCTAssertLessThan(Date().timeIntervalSince(started), 10)
        XCTAssertEqual(result.status, 0)
        XCTAssertEqual(result.stdout, "done\n")
        XCTAssertEqual(result.stderr.count, 1_048_576)
    }
}
