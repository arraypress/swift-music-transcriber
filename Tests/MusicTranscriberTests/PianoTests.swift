//
//  PianoTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  The piano engine against upstream's own numbers on a real piano loop:
//  the front end, the GRU arithmetic, the trunk through Core AI, one segment's
//  seven outputs, the post-processor on upstream's outputs, and the events end
//  to end. Model-backed tests skip when no piano asset is installed.
//

import XCTest
@testable import MusicTranscriber

final class PianoTests: XCTestCase {

    struct Events: Decodable {
        struct Note: Decodable { let onset_time: Float; let offset_time: Float; let midi_note: Int; let velocity: Int }
        struct Pedal: Decodable { let onset_time: Float; let offset_time: Float }
        let notes: [Note]; let pedals: [Pedal]; let audio_len: Int
    }

    private func floats(_ name: String) throws -> [Float] { try Fixture.floats("Piano/\(name)") }

    private func upstreamOutputs() throws -> PianoPostprocessing.Outputs {
        PianoPostprocessing.Outputs(frames: 1001,
                                    regOnset: try floats("reg_onset_output.f32"), regOffset: try floats("reg_offset_output.f32"),
                                    frame: try floats("frame_output.f32"), velocity: try floats("velocity_output.f32"),
                                    pedalOnset: try floats("reg_pedal_onset_output.f32"), pedalOffset: try floats("reg_pedal_offset_output.f32"),
                                    pedalFrame: try floats("pedal_frame_output.f32"))
    }

    private func loadPiano() async throws -> PianoTranscriber {
        let url: URL
        do { url = try ModelLocator.resolvePiano() } catch { throw XCTSkip("no piano asset installed: \(error)") }
        return try await PianoTranscriber(contentsOf: url)
    }

    private func segmentSamples() throws -> [Float] {
        var samples = try AudioLoader.load(try Fixture.url("Piano/audio_16k.wav"))
        XCTAssertLessThanOrEqual(samples.count, PianoFrontEnd.segmentSamples)
        samples.append(contentsOf: [Float](repeating: 0, count: PianoFrontEnd.segmentSamples - samples.count))
        return samples
    }

    // MARK: - No model needed

    func testFrontEndMatchesUpstream() throws {
        let reference = try floats("logmel_0.f32")
        let ours = PianoFrontEnd().logMel(segment: try segmentSamples())
        XCTAssertEqual(ours.count, reference.count)
        let psnr = Fixture.psnr(reference, ours)
        XCTAssertGreaterThan(psnr, 90, "PSNR \(psnr) dB")
    }

    func testGRUMatchesPyTorch() throws {
        let p = try floats("tiny_gru_parameters.f32")
        var cursor = 0
        let head = GRUHead.parse(p, cursor: &cursor, layers: 2, inputSize: 16, hidden: 8, classes: 3)
        XCTAssertEqual(cursor, p.count)
        let out = head.run(try floats("tiny_gru_input.f32"), frames: 37)
        let reference = try floats("tiny_gru_output.f32")
        var worst: Float = 0
        for i in out.indices { worst = max(worst, abs(out[i] - reference[i])) }
        XCTAssertLessThan(worst, 1e-5, "max |diff| \(worst)")
    }

    func testPostprocessorReproducesUpstreamEvents() throws {
        let expected = try Fixture.json("Piano/events.json", as: Events.self)
        let (notes, pedals) = PianoPostprocessing.events(from: try upstreamOutputs())
        XCTAssertEqual(notes.count, expected.notes.count)
        XCTAssertEqual(pedals.count, expected.pedals.count)
        for (n, e) in zip(notes, expected.notes) {
            XCTAssertEqual(n.midiNote, e.midi_note); XCTAssertEqual(n.velocity, e.velocity)
            XCTAssertEqual(n.onset, e.onset_time, accuracy: 1e-6); XCTAssertEqual(n.offset, e.offset_time, accuracy: 1e-6)
        }
        for (p, e) in zip(pedals, expected.pedals) {
            XCTAssertEqual(p.onset, e.onset_time, accuracy: 1e-6); XCTAssertEqual(p.offset, e.offset_time, accuracy: 1e-6)
        }
    }

    // MARK: - Model

    func testTrunkMatchesUpstream() async throws {
        let piano = try await loadPiano()
        let trunks = try await piano.trunkFeatures(mel: try floats("logmel_0.f32"))
        let reference = try floats("frame_trunk_0.f32")
        let psnr = Fixture.psnr(reference, trunks[0])
        print("PIANO trunk PSNR \(psnr) dB")
        XCTAssertGreaterThan(psnr, 60, "trunk PSNR \(psnr) dB")
    }

    func testBranchGRUMatchesUpstream() async throws {
        let piano = try await loadPiano()
        // Upstream's own trunk output through our GRU: isolates the recurrence from the GPU.
        let feats = piano.heads[0].features(try floats("frame_trunk_0.f32"), frames: 1001)
        let reference = try floats("frame_gru_0.f32")
        let psnr = Fixture.psnr(reference, feats)
        print("PIANO GRU PSNR \(psnr) dB")
        XCTAssertGreaterThan(psnr, 80, "GRU PSNR \(psnr) dB")
    }

    func testSegmentOutputsMatchUpstream() async throws {
        let piano = try await loadPiano()
        let ours = try await piano.segmentOutputs(mel: try floats("logmel_0.f32"))
        for key in ["reg_onset_output", "reg_offset_output", "frame_output", "velocity_output", "reg_pedal_onset_output", "reg_pedal_offset_output", "pedal_frame_output"] {
            let psnr = Fixture.psnr(try floats("\(key).f32"), ours[key]!)
            print("PIANO \(key) PSNR \(psnr) dB")
            XCTAssertGreaterThan(psnr, 50, "\(key) PSNR \(psnr) dB")
        }
    }

    func testEventsMatchUpstream() async throws {
        try await compareEvents(audio: "Piano/audio_16k.wav", events: "Piano/events.json", label: "one segment")
    }

    /// Seventeen seconds: three overlapping segments, so the stitching is exercised too.
    func testStitchedEventsMatchUpstream() async throws {
        try await compareEvents(audio: "Piano/render_audio_16k.wav", events: "Piano/render_events.json", label: "three segments")
    }

    private func compareEvents(audio: String, events: String, label: String) async throws {
        let piano = try await loadPiano()
        let expected = try Fixture.json(events, as: Events.self)
        let samples = try AudioLoader.load(try Fixture.url(audio))
        XCTAssertEqual(samples.count, expected.audio_len)
        let (notes, pedals) = try await piano.transcribe(samples16k: samples)
        print("PIANO events (\(label)): ours \(notes.count) notes / \(pedals.count) pedals, upstream \(expected.notes.count) / \(expected.pedals.count)")
        XCTAssertEqual(notes.count, expected.notes.count)
        XCTAssertEqual(pedals.count, expected.pedals.count)
        var worst: Float = 0
        for (n, e) in zip(notes, expected.notes) {
            XCTAssertEqual(n.midiNote, e.midi_note); XCTAssertEqual(n.velocity, e.velocity, "velocity of note \(e.midi_note) at \(e.onset_time)")
            worst = max(worst, abs(n.onset - e.onset_time), abs(n.offset - e.offset_time))
        }
        for (p, e) in zip(pedals, expected.pedals) { worst = max(worst, abs(p.onset - e.onset_time), abs(p.offset - e.offset_time)) }
        print("PIANO worst timing difference \(worst * 1000) ms")
        XCTAssertLessThan(worst, 0.001)
    }
}
