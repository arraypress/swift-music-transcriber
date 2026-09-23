//
//  LogitParityTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  How close the Core AI decoder's logits are to upstream's CPU fp32 logits on
//  the same forced token sequence — and, where the argmax differs, how close
//  the two top logits were. Skipped unless SCRIBE_LOGIT_CASES lists cases as
//  "size:<reference clip dir>:<chunk>;…" produced by reference_tokens.py plus
//  dump_logits.py.
//

import XCTest
@testable import MusicTranscriber

final class LogitParityTests: XCTestCase {

    struct Reference: Decodable {
        struct Chunk: Decodable { let prompt: [Int]; let tokens: [Int] }
        let chunks: [Chunk]
    }

    func testForcedLogitsAgainstUpstream() async throws {
        guard let spec = ProcessInfo.processInfo.environment["SCRIBE_LOGIT_CASES"] else {
            throw XCTSkip("set SCRIBE_LOGIT_CASES")
        }
        var loaded: [String: CoreAIDecoder] = [:]
        for item in spec.split(separator: ";") {
            let parts = item.split(separator: ":").map(String.init)
            let size = parts[0], dir = parts[1], chunkIndex = Int(parts[2])!
            let fixtures = "\(dir)/\(size)/fixtures"
            let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: URL(fileURLWithPath: "\(fixtures)/chunks.json")))
            let chunk = reference.chunks[chunkIndex]
            let mel = try Data(contentsOf: URL(fileURLWithPath: "\(fixtures)/mel_\(chunkIndex).f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let theirs = try Data(contentsOf: URL(fileURLWithPath: "\(fixtures)/logits_\(chunkIndex).f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

            if loaded[size] == nil {
                loaded[size] = try await CoreAIDecoder(contentsOf: try ModelLocator.resolve(variant: ModelVariant(rawValue: size)!, precision: .float32))
            }
            let decoder = loaded[size]!
            let card = decoder.card
            let sequence = chunk.prompt + chunk.tokens
            XCTAssertEqual(theirs.count, (sequence.count + 1) * card)

            // Pass 1: upstream's own mel. Pass 2: our mel from the very samples upstream decoded.
            let samples = try AudioLoader.load(URL(fileURLWithPath: "\(dir)/audio_16k.wav"))
            let ours = MelSpectrogram().logMel(Array(AudioLoader.chunks(samples)[chunkIndex]))
            let clip = URL(fileURLWithPath: dir).lastPathComponent
            try ours.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: "\(fixtures)/mel_ours_\(chunkIndex).f32"))
            print(String(format: "MEL %@ chunk %d: ours vs upstream %.1f dB", clip, chunkIndex, Fixture.psnr(mel, ours)))
            for (label, input) in [("upstream mel", mel), ("our mel", ours)] {
                let psnrs = try await force(decoder: decoder, mel: input, sequence: sequence, reference: theirs, label: "\(clip) \(size) chunk \(chunkIndex) [\(label)]")
                XCTAssertGreaterThan(psnrs, 40, "\(dir) \(size) [\(label)]: logits drifted")
            }
        }
    }

    /// Teacher-forces `sequence` after the prefix, as dump_logits.py did, and prints per-step agreement plus the top-two margin at every argmax flip. Returns the minimum PSNR.
    private func force(decoder: CoreAIDecoder, mel: [Float], sequence: [Int], reference theirs: [Float], label: String) async throws -> Double {
        let card = decoder.card
        let state = decoder.makeState()
        let prefix = try await decoder.prefixEmbeddings(mel: mel, frames: MelSpectrogram.frames, instrumentTokens: [0])
        var length = prefix.count + 1
        var logits = try await decoder.logits(embeddings: prefix.values + (try await decoder.embeddings(of: [Int32(card)])),
                                              count: length, sequenceLength: length, state: state)
        var psnrs: [Double] = []
        var flips: [String] = []
        for step in 0...sequence.count {
            let ref = Array(theirs[(step * card)..<((step + 1) * card)])
            psnrs.append(Fixture.psnr(ref, logits))
            let a = Sampling.argmax(ref), b = Sampling.argmax(logits)
            if a != b {
                let sortedRef = ref.sorted(by: >), sortedOurs = logits.sorted(by: >)
                flips.append(String(format: "step %d: upstream %d over %d by %.4f, ours %d over %d by %.4f",
                                    step, a, b, sortedRef[0] - sortedRef[1], b, a, sortedOurs[0] - sortedOurs[1]))
            }
            guard step < sequence.count else { break }
            length += 1
            logits = try await decoder.logits(embeddings: try await decoder.embeddings(of: [Int32(sequence[step])]),
                                              count: 1, sequenceLength: length, state: state)
        }
        let sorted = psnrs.sorted()
        print(String(format: "LOGITS %@: %d steps, PSNR min %.1f median %.1f dB; argmax flips %d%@",
                     label, psnrs.count, sorted[0], sorted[sorted.count / 2], flips.count,
                     flips.isEmpty ? "" : " — " + flips.joined(separator: "; ")))
        return sorted[0]
    }
}
