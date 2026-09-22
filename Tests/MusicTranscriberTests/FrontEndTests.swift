//
//  FrontEndTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  The mel front end and the resampler against torchaudio and julius.
//

import XCTest
@testable import MusicTranscriber

final class FrontEndTests: XCTestCase {

    func testFilterbankMatchesTorchaudio() throws {
        let reference = try Fixture.floats("mel_filterbank.f32")
        let ours = MelSpectrogram.htkFilterbank()
        XCTAssertEqual(ours.count, reference.count)
        var worst: Float = 0
        for i in 0..<ours.count { worst = max(worst, abs(ours[i] - reference[i])) }
        // torchaudio builds its bank in float32; ours is computed in double.
        // The log-mel test below is the one that matters and holds above 90 dB.
        XCTAssertLessThan(worst, 1e-4, "max |diff| \(worst)")
    }

    func testLogMelMatchesTorchaudio() throws {
        let input = try Fixture.floats("mel_input.f32")
        let reference = try Fixture.floats("mel_output.f32")
        XCTAssertEqual(input.count, MelSpectrogram.chunkSamples)
        let ours = MelSpectrogram().logMel(input)
        XCTAssertEqual(ours.count, reference.count)
        let psnr = Fixture.psnr(reference, ours)
        var worst: Float = 0
        for i in 0..<ours.count { worst = max(worst, abs(ours[i] - reference[i])) }
        XCTAssertGreaterThan(psnr, 90, "PSNR \(psnr) dB, max |diff| \(worst)")
    }

    func testResamplerMatchesJulius() throws {
        for rate in [44100, 48000] {
            let input = try Fixture.floats("resample_\(rate)_in.f32")
            let reference = try Fixture.floats("resample_\(rate)_out.f32")
            let ours = Resampler(from: rate, to: 16000).resample(input)
            XCTAssertEqual(ours.count, reference.count, "\(rate)")
            let psnr = Fixture.psnr(reference, ours)
            XCTAssertGreaterThan(psnr, 90, "\(rate): PSNR \(psnr) dB")
        }
    }

    func testSameRateIsIdentity() {
        let x: [Float] = [0.1, 0.2, 0.3]
        XCTAssertEqual(Resampler(from: 16000, to: 16000).resample(x), x)
    }
}
