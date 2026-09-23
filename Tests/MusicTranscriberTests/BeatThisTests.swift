//
//  BeatThisTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  The Beat This! port against upstream: front end, chunking, post-processing
//  as pure functions on fixtures; the model itself and the whole tracker when
//  the asset is installed.
//

import XCTest
@testable import MusicTranscriber

final class BeatThisTests: XCTestCase {

    struct Beats: Decodable { let frames: Int; let samples16: Int; let samples22: Int; let beats: [Double]; let downbeats: [Double] }

    private func floats(_ name: String) throws -> [Float] { try Fixture.floats("BeatThis/\(name)") }
    private func beats(_ clip: String) throws -> Beats { try Fixture.json("BeatThis/\(clip)_beats.json", as: Beats.self) }

    func testFilterbankMatchesTorchaudio() throws {
        let reference = try floats("mel_filterbank.f32")
        let ours = BeatThisFrontEnd.slaneyFilterbank()
        XCTAssertEqual(ours.count, reference.count)
        var worst: Float = 0
        for i in ours.indices { worst = max(worst, abs(ours[i] - reference[i])) }
        XCTAssertLessThan(worst, 1e-4, "max |diff| \(worst)")
    }

    func testLogMelMatchesTorchaudio() throws {
        let signal = try floats("wav22.f32")
        let reference = try floats("mel.f32")
        let (ours, frames) = BeatThisFrontEnd().logMel(signal)
        XCTAssertEqual(frames, try beats("demo").frames)
        XCTAssertEqual(ours.count, reference.count)
        let psnr = Fixture.psnr(reference, ours)
        XCTAssertGreaterThan(psnr, 80, "PSNR \(psnr) dB")
    }

    func testChunkingMatchesUpstream() {
        let short = BeatPostprocessing.chunks(frames: 502)
        XCTAssertEqual(short, [.init(start: -6, from: 0, to: 502, padLeft: 6, padRight: 6)])
        let long = BeatPostprocessing.chunks(frames: 6103)
        XCTAssertEqual(long.map(\.start), [-6, 1482, 2970, 4458, 4609], "upstream's split_piece with the last chunk shifted to the end")
        XCTAssertEqual(long.first?.padLeft, 6)
        XCTAssertEqual(long.last?.padRight, 6)
        XCTAssertEqual(long.last!.to - long.last!.from, 1494)
    }

    /// Peak picking, deduplication and downbeat snapping against upstream's
    /// own beats for four clips, from its own logits: identical times.
    func testPostprocessingMatchesUpstream() throws {
        for clip in ["demo", "makina", "piano", "bass"] {
            let expected = try beats(clip)
            let got = BeatPostprocessing.beats(beatLogits: try floats("\(clip)_beat_logits.f32"),
                                               downbeatLogits: try floats("\(clip)_downbeat_logits.f32"))
            XCTAssertEqual(got.beats.count, expected.beats.count, clip)
            XCTAssertEqual(got.downbeats.count, expected.downbeats.count, clip)
            for (a, b) in zip(got.beats, expected.beats) { XCTAssertEqual(a, b, accuracy: 1e-9, clip) }
            for (a, b) in zip(got.downbeats, expected.downbeats) { XCTAssertEqual(a, b, accuracy: 1e-9, clip) }
        }
    }

    /// The exported model against PyTorch on the demo's padded chunk.
    func testModelMatchesPyTorch() async throws {
        guard let url = try? ModelLocator.resolveBeatTracker() else { throw XCTSkip("no beat tracker installed") }
        let tracker = try await BeatThisTracker(contentsOf: url)
        let input = try floats("chunk_input.f32")
        let frames = input.count / BeatThisFrontEnd.mels
        // The fixture is upstream's own padded chunk and the logits it got for it:
        // one pass, no further chunking, so the two runs see the same context.
        let (beat, downbeat) = try await tracker.run(chunk: input, rows: frames)
        XCTAssertEqual(beat.count, frames)
        let refBeat = try floats("chunk_beat.f32"), refDown = try floats("chunk_downbeat.f32")
        let psnrBeat = Fixture.psnr(refBeat, beat)
        let psnrDown = Fixture.psnr(refDown, downbeat)
        XCTAssertGreaterThan(psnrBeat, 40, "beat logits PSNR \(psnrBeat) dB")
        XCTAssertGreaterThan(psnrDown, 40, "downbeat logits PSNR \(psnrDown) dB")
    }

    /// Whole tracker from the 16 kHz signal, as MuScriptor calls it. The only
    /// non-shared code on the way is the 16→22.05 kHz resampler (julius here,
    /// soxr upstream), so agreement is measured, not asserted exact.
    func testTrackerAgreesWithUpstream() async throws {
        guard let url = try? ModelLocator.resolveBeatTracker() else { throw XCTSkip("no beat tracker installed") }
        let tracker = try await BeatThisTracker(contentsOf: url)
        for clip in ["demo", "piano", "bass"] {
            let expected = try beats(clip)
            let got = try await tracker.track(samples16k: try floats("\(clip)_wav16.f32"))
            let matched = got.beats.filter { g in expected.beats.contains { abs($0 - g) <= 0.02 } }.count
            print("BEATS \(clip): ours \(got.beats.count) beats / \(got.downbeats.count) downbeats, upstream \(expected.beats.count) / \(expected.downbeats.count); \(matched) beats within one frame")
            XCTAssertEqual(got.beats.count, expected.beats.count, clip)
            XCTAssertGreaterThanOrEqual(matched, expected.beats.count - 1, "\(clip): beats within one frame of upstream's")
        }
    }
}
