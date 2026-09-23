//
//  ParityTests.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  Token-for-token agreement with upstream over a whole directory of clips.
//  Skipped unless SCRIBE_PARITY_DIR points at Tools/reference_tokens.py output:
//
//    <dir>/<clip>/audio_16k.wav                 the exact samples upstream decoded
//    <dir>/<clip>/<size>/fixtures/chunks.json   upstream's prompts and tokens per chunk
//
//  Every clip × size is a separate check; the summary is printed at the end.
//

import XCTest
@testable import MusicTranscriber

final class ParityTests: XCTestCase {

    struct Reference: Decodable {
        struct Chunk: Decodable { let prompt: [Int]; let tokens: [Int] }
        let size: String; let instruments: [String]; let chunks: [Chunk]
    }

    func testEveryReferenceClip() async throws {
        guard let root = ProcessInfo.processInfo.environment["SCRIBE_PARITY_DIR"] else {
            throw XCTSkip("set SCRIBE_PARITY_DIR to a reference_tokens.py output directory")
        }
        let fm = FileManager.default
        let clips = try fm.contentsOfDirectory(atPath: root).sorted()
            .filter { fm.fileExists(atPath: "\(root)/\($0)/audio_16k.wav") }
        var summary: [String] = []
        var loaded: [String: MusicTranscriber] = [:]

        for clip in clips {
            let audio = URL(fileURLWithPath: "\(root)/\(clip)/audio_16k.wav")
            for size in ["small", "medium", "large"] {
                let path = "\(root)/\(clip)/\(size)/fixtures/chunks.json"
                guard fm.fileExists(atPath: path) else { continue }
                let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
                let variant = ModelVariant(rawValue: size)!
                if loaded[size] == nil {
                    guard let model = try? ModelLocator.resolve(variant: variant, precision: .float32) else {
                        summary.append("\(clip) \(size): NO MODEL"); continue
                    }
                    loaded[size] = try await MusicTranscriber(model: model)
                }
                let transcriber = loaded[size]!
                let box = DiagnosticTests.Box()
                transcriber.tokenObserver = { chunk, prompt, tokens in box.record(chunk, prompt, tokens) }
                var options = TranscriptionOptions()
                options.instruments = try InstrumentGroup.resolve(reference.instruments)
                _ = try await transcriber.transcribe(audio, options: options, tempo: .off)

                var mismatches: [String] = []
                for (i, expected) in reference.chunks.enumerated() {
                    guard let (prompt, tokens) = box.chunks[i] else { mismatches.append("chunk \(i) missing"); continue }
                    if prompt != expected.prompt { mismatches.append("chunk \(i) prompt differs") }
                    if tokens != expected.tokens {
                        let first = zip(tokens, expected.tokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                            ?? min(tokens.count, expected.tokens.count)
                        mismatches.append("chunk \(i): first diff at token \(first) of \(expected.tokens.count) (ours \(tokens.count))")
                    }
                }
                let tokens = reference.chunks.reduce(0) { $0 + $1.tokens.count }
                let label = "\(clip) \(size)\(reference.instruments.isEmpty ? "" : " [\(reference.instruments.joined(separator: ","))]")"
                if mismatches.isEmpty {
                    summary.append("PASS \(label): \(reference.chunks.count) chunks, \(tokens) tokens identical")
                } else {
                    // Not identical: how far apart are the two transcriptions as notes?
                    let theirs = Self.notes(from: reference.chunks.map { ($0.prompt, $0.tokens) })
                    let ours = Self.notes(from: reference.chunks.indices.map { box.chunks[$0] ?? ([], []) })
                    let f1 = Self.noteF1(ours, theirs)
                    summary.append(String(format: "DIFF %@: %@ — notes ours %d / upstream %d, onset F1 %.3f",
                                          label, mismatches.joined(separator: "; "), ours.count, theirs.count, f1))
                }
            }
        }
        print("PARITY SUMMARY\n" + summary.joined(separator: "\n"))
        XCTAssertFalse(summary.isEmpty, "no references found under \(root)")
    }

    /// Notes from per-chunk (prompt, tokens) through the same decoder and cleanup.
    static func notes(from chunks: [([Int], [Int])]) -> [TranscribedNote] {
        var decoder = TokenDecoder()
        var events: [TranscriptionEvent] = []
        for (i, chunk) in chunks.enumerated() {
            events += decoder.beginChunk(seekTime: Double(i) * 5, nextSeekTime: i + 1 < chunks.count ? Double(i + 1) * 5 : nil)
            for token in chunk.0 + chunk.1 { events += decoder.feed(token) }
        }
        events += decoder.finish()
        return NoteCleanup.cleaned(NoteCleanup.notes(from: events))
    }

    /// mir_eval-style onset F1: same pitch, same program, onsets within 50 ms, one-to-one.
    static func noteF1(_ a: [TranscribedNote], _ b: [TranscribedNote]) -> Double {
        var used = Set<Int>()
        var hits = 0
        for n in a {
            if let j = b.indices.first(where: { !used.contains($0) && b[$0].pitch == n.pitch && b[$0].program == n.program && abs(b[$0].onset - n.onset) <= 0.05 }) {
                used.insert(j); hits += 1
            }
        }
        guard !a.isEmpty, !b.isEmpty else { return a.isEmpty && b.isEmpty ? 1 : 0 }
        let p = Double(hits) / Double(a.count), r = Double(hits) / Double(b.count)
        return p + r == 0 ? 0 : 2 * p * r / (p + r)
    }
}
