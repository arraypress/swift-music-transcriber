//
//  InstrumentGroupTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//

import XCTest
@testable import MusicTranscriber

final class InstrumentGroupTests: XCTestCase {

    func testTableShape() {
        XCTAssertEqual(InstrumentGroup.all.count, 35, "34 pitched groups and drums; the two FX classes have no name upstream")
        XCTAssertEqual(InstrumentGroup.named("drums")?.id, 36)
        XCTAssertEqual(InstrumentGroup.named("electric_piano")?.representativeProgram, 2)
        XCTAssertEqual(InstrumentGroup.label(forProgram: 33), "electric_bass")
        XCTAssertEqual(InstrumentGroup.label(forProgram: 128), "drums")
        XCTAssertEqual(InstrumentGroup.label(forProgram: 96), "program_96")
    }

    func testResolve() throws {
        XCTAssertEqual(try InstrumentGroup.resolve(["Timp", "cello", "dist"]).map(\.name),
                       ["timpani", "cello", "distorted_electric_guitar"])
        XCTAssertThrowsError(try InstrumentGroup.resolve(["guitar"])) { error in
            guard case MusicTranscriberError.ambiguousInstrument(_, let candidates) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(candidates.count, 3)
        }
        XCTAssertThrowsError(try InstrumentGroup.resolve(["pinao"])) { error in
            guard case MusicTranscriberError.unknownInstrument(_, let suggestions) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(suggestions.contains("acoustic_piano"), "\(suggestions)")
        }
    }
}
