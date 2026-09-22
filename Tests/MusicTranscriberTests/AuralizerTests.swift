//
//  AuralizerTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//

import MIDIFileKit
import XCTest
@testable import MusicTranscriber

final class AuralizerTests: XCTestCase {

    /// The built-in synthesiser renders the reference MIDI to something audible
    /// of the right length, offline.
    func testSynthesizesReferenceMIDI() throws {
        let midi = try Data(contentsOf: try Fixture.url("reference.mid"))
        let samples = try Auralizer.synthesize(midi: midi)
        // The last note of the reference ends at 14.89 s: the model's final chunk
        // has no window to clip to, so a note-off can land well past the audio.
        let expected = try MIDIReader.read(midi).duration + 2
        XCTAssertEqual(Double(samples.count) / Auralizer.sampleRate, expected, accuracy: 0.1)
        XCTAssertGreaterThan(Auralizer.rms(samples), 0.001, "silence means the synth never received the notes")
    }

    func testRenderWritesStereoWAV() throws {
        let midi = try Data(contentsOf: try Fixture.url("reference.mid"))
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("scribe-auralize-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: out) }
        try Auralizer.render(midi: midi, original: try Fixture.url("demo_16k.wav"), to: out)
        let left = try AudioLoader.load(out, sampleRate: 44_100)
        XCTAssertGreaterThan(left.count, Int(10 * 44_100))
    }
}
