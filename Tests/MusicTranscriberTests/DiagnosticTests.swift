//
//  DiagnosticTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  Where exactly a real-model run parts from upstream, chunk by chunk.
//

import XCTest
@testable import MusicTranscriber

final class DiagnosticTests: XCTestCase {

    func testWhereTokensDiverge() async throws {
        guard let model = try? ModelLocator.resolve(variant: .medium, precision: .float32) else {
            throw XCTSkip("no medium model installed")
        }
        let file = try Fixture.json("decoder.json", as: TokenDecoderTests.File.self)
        let transcriber = try await MusicTranscriber(model: model)
        let box = Box()
        transcriber.tokenObserver = { chunk, prompt, tokens in box.record(chunk, prompt, tokens) }
        _ = try await transcriber.transcribe(try Fixture.url("demo_16k.wav"), tempo: .off)

        for (i, expected) in file.chunks.enumerated() {
            guard let (prompt, tokens) = box.chunks[i] else { print("chunk \(i): not generated"); continue }
            let firstPrompt = zip(prompt, expected.prompt).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            let first = zip(tokens, expected.tokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? (tokens.count == expected.tokens.count ? nil : min(tokens.count, expected.tokens.count))
            print("chunk \(i): prompt \(prompt.count) vs \(expected.prompt.count) (first diff \(firstPrompt.map(String.init) ?? "none")), "
                  + "tokens \(tokens.count) vs \(expected.tokens.count), first diff at \(first.map(String.init) ?? "none")")
            if let first, first < tokens.count, first < expected.tokens.count {
                print("   ours: \(tokens[max(0, first - 3)..<min(tokens.count, first + 4)].map { "\(EventVocabulary.event($0))" })")
                print("   theirs: \(expected.tokens[max(0, first - 3)..<min(expected.tokens.count, first + 4)].map { "\(EventVocabulary.event($0))" })")
            }
        }
        // And how far our mel is from torch's on chunk 0 of the same samples.
        let samples = try AudioLoader.load(try Fixture.url("demo_16k.wav"))
        print("samples: \(samples.count), chunks: \(AudioLoader.chunks(samples).map(\.count))")
        let ours = MelSpectrogram().logMel(Array(samples[0..<80000]))
        let reference = try Fixture.floats("mel_output.f32")
        var worst: Float = 0
        for i in 0..<ours.count { worst = max(worst, abs(ours[i] - reference[i])) }
        print("mel chunk 0: PSNR \(Fixture.psnr(reference, ours)) dB, max |diff| \(worst)")
    }

    final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var _chunks: [Int: ([Int], [Int])] = [:]
        var chunks: [Int: ([Int], [Int])] { lock.withLock { _chunks } }
        func record(_ chunk: Int, _ prompt: [Int], _ tokens: [Int]) { lock.withLock { _chunks[chunk] = (prompt, tokens) } }
    }
}
