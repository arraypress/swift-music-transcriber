//
//  ChunkGenerator.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The generation loop for one 5-second chunk: greedy, sampled, guided or
//  beam-searched, exactly as upstream's LMModel.generate behaves.
//

import Foundation

/// Generates one chunk's tokens.
struct ChunkGenerator {

    let decoder: CoreAIDecoder
    let options: TranscriptionOptions
    let forbidden: [Int]
    let instrumentTokens: [Int32]

    /// What a chunk produced.
    struct Result {
        /// Generated tokens, the end token excluded.
        var tokens: [Int]
        /// Whether the model ended the chunk itself.
        var ended: Bool
    }

    /// Conditioning for one chunk, built once and reused by every pass.
    struct Conditioning {
        let prefix: [Float]
        let prefixCount: Int
        let nullPrefix: [Float]?
    }

    /// Build the prefix (and the null prefix when guidance is on).
    func conditioning(mel: [Float]) async throws -> Conditioning {
        let prefix = try await decoder.prefixEmbeddings(mel: mel, frames: MelSpectrogram.frames,
                                                        instrumentTokens: instrumentTokens)
        var nullPrefix: [Float]?
        if options.guidance != 1.0 {
            // Upstream nulls every condition: the wav becomes a zero-length signal
            // whose 501 mel frames are then masked to zero, and the class tokens
            // become the pad class. Same length as the real prefix, all zeros
            // where the mel was.
            let null = try await decoder.prefixEmbeddings(mel: mel, frames: MelSpectrogram.frames, instrumentTokens: [0])
            var values = null.values
            for i in 0..<(MelSpectrogram.frames * decoder.dimension) { values[i] = 0 }
            nullPrefix = values
        }
        return Conditioning(prefix: prefix.values, prefixCount: prefix.count, nullPrefix: nullPrefix)
    }

    /// Decode a chunk. `prompt` is the teacher-forced tie prologue, if any.
    func generate(conditioning: Conditioning, prompt: [Int], initialToken: Int,
                  states: [CoreAIDecoder.State], generator: inout SeededGenerator?) async throws -> Result {
        if options.beamSize > 1 {
            return try await beamSearch(conditioning: conditioning, prompt: prompt, initialToken: initialToken, states: states)
        }
        return try await sequential(conditioning: conditioning, prompt: prompt, initialToken: initialToken,
                                    states: states, generator: &generator)
    }

    // MARK: - Greedy / sampling

    private func sequential(conditioning: Conditioning, prompt: [Int], initialToken: Int,
                            states: [CoreAIDecoder.State], generator: inout SeededGenerator?) async throws -> Result {
        let budget = options.maximumTokens - prompt.count
        let guided = conditioning.nullPrefix != nil
        let state = states[0]
        decoder.reset(state)
        let nullState: CoreAIDecoder.State? = guided ? states[1] : nil
        if let nullState { decoder.reset(nullState) }

        let promptTokens = ([initialToken] + prompt).map(Int32.init)
        let tokenEmbeds = try await decoder.embeddings(of: promptTokens)
        var sequenceLength = conditioning.prefixCount + promptTokens.count
        var logits = try await decoder.logits(embeddings: conditioning.prefix + tokenEmbeds,
                                              count: sequenceLength, sequenceLength: sequenceLength, state: state)
        if let nullState, let nullPrefix = conditioning.nullPrefix {
            let uncond = try await decoder.logits(embeddings: nullPrefix + tokenEmbeds,
                                                  count: sequenceLength, sequenceLength: sequenceLength, state: nullState)
            logits = Self.guide(conditional: logits, unconditional: uncond, coefficient: Float(options.guidance))
        }

        var tokens: [Int] = []
        var ended = false
        for _ in 0..<max(0, budget) {
            Sampling.mask(&logits, forbidden: forbidden)
            let next: Int
            if options.sampling, options.temperature > 0 {
                if generator != nil {
                    next = Sampling.sample(logits, temperature: options.temperature, using: &generator!)
                } else {
                    var system = SystemRandomNumberGenerator()
                    next = Sampling.sample(logits, temperature: options.temperature, using: &system)
                }
            } else {
                next = Sampling.argmax(logits)
            }
            if next == EventVocabulary.endOfChunk { ended = true; break }
            tokens.append(next)
            // Upstream never feeds the last budgeted token back in; doing so would
            // also be one position past the cache.
            if tokens.count == budget { break }
            let embeds = try await decoder.embeddings(of: [Int32(next)])
            sequenceLength += 1
            logits = try await decoder.logits(embeddings: embeds, count: 1, sequenceLength: sequenceLength, state: state)
            if let nullState {
                let uncond = try await decoder.logits(embeddings: embeds, count: 1, sequenceLength: sequenceLength, state: nullState)
                logits = Self.guide(conditional: logits, unconditional: uncond, coefficient: Float(options.guidance))
            }
        }
        return Result(tokens: tokens, ended: ended)
    }

    static func guide(conditional: [Float], unconditional: [Float], coefficient: Float) -> [Float] {
        var out = conditional
        for i in out.indices { out[i] = unconditional[i] + (conditional[i] - unconditional[i]) * coefficient }
        return out
    }

    // MARK: - Beam search

    /// Upstream's length-normalised beam search (alpha 0.75), one cache per beam.
    ///
    /// Each step scores every beam's top `beamSize` continuations, normalised by
    /// `length^0.75`, keeps the best `beamSize` overall, and copies the winners'
    /// caches into place. Ended beams are frozen but carried along, as upstream
    /// does, and the best beam's tokens up to its end token are returned.
    private func beamSearch(conditioning: Conditioning, prompt: [Int], initialToken: Int,
                            states: [CoreAIDecoder.State]) async throws -> Result {
        let width = options.beamSize
        let alpha = 0.75
        let guided = conditioning.nullPrefix != nil
        precondition(states.count >= width * (guided ? 2 : 1) + width, "beam search needs states for every beam plus scratch")
        let beamStates = Array(states[0..<width])
        let nullStates = guided ? Array(states[width..<(2 * width)]) : []
        let scratch = states[(guided ? 2 * width : width)...]
        for s in states { decoder.reset(s) }

        let promptTokens = ([initialToken] + prompt).map(Int32.init)
        let tokenEmbeds = try await decoder.embeddings(of: promptTokens)
        var sequenceLength = conditioning.prefixCount + promptTokens.count
        var sequences = [[Int]](repeating: [initialToken] + prompt, count: width)
        var scores = [Double](repeating: 0, count: width)
        var logitsPerBeam: [[Float]] = []

        // Prefill every beam identically; the first step only expands beam 0.
        for b in 0..<width {
            var l = try await decoder.logits(embeddings: conditioning.prefix + tokenEmbeds,
                                             count: sequenceLength, sequenceLength: sequenceLength, state: beamStates[b])
            if guided, let nullPrefix = conditioning.nullPrefix {
                let u = try await decoder.logits(embeddings: nullPrefix + tokenEmbeds,
                                                 count: sequenceLength, sequenceLength: sequenceLength, state: nullStates[b])
                l = Self.guide(conditional: l, unconditional: u, coefficient: Float(options.guidance))
            }
            logitsPerBeam.append(l)
        }

        let budget = options.maximumTokens - prompt.count
        var lastEnded = false
        for step in 0..<max(0, budget) {
            struct Candidate { let parent: Int; let token: Int; let normalised: Double; let raw: Double }
            var candidates: [Candidate] = []
            for b in 0..<width {
                var logits = logitsPerBeam[b]
                Sampling.mask(&logits, forbidden: forbidden)
                let logProbs = Sampling.logSoftmax(logits)
                let ended = sequences[b].contains(EventVocabulary.endOfChunk)
                let endPosition = sequences[b].firstIndex(of: EventVocabulary.endOfChunk) ?? 0
                let length = Double(ended ? max(endPosition, 1) : sequences[b].count)
                let lp = 1 / pow(length, alpha)
                let top = logProbs.enumerated().sorted { $0.element > $1.element }.prefix(width)
                for (token, score) in top {
                    let s = ended ? 0 : score
                    candidates.append(Candidate(parent: b, token: token, normalised: (scores[b] + s) * lp, raw: scores[b] + s))
                }
                if step == 0 { break }   // all beams identical at the start: take beam 0's top-k
            }
            let chosen: [Candidate] = step == 0
                ? Array(candidates.prefix(width))
                : Array(candidates.sorted { $0.normalised > $1.normalised }.prefix(width))

            // Reorder: build the new beams in scratch, then swap in.
            var newSequences: [[Int]] = []
            var newScores: [Double] = []
            for (i, c) in chosen.enumerated() {
                decoder.copy(beamStates[c.parent], into: scratch[scratch.startIndex + i])
                if guided { decoder.copy(nullStates[c.parent], into: scratch[scratch.startIndex + width + i]) }
                newSequences.append(sequences[c.parent] + [c.token])
                newScores.append(c.raw)
            }
            for i in 0..<width {
                decoder.copy(scratch[scratch.startIndex + i], into: beamStates[i])
                if guided { decoder.copy(scratch[scratch.startIndex + width + i], into: nullStates[i]) }
            }
            sequences = newSequences
            scores = newScores
            sequenceLength += 1

            if sequences.allSatisfy({ $0.contains(EventVocabulary.endOfChunk) }) { lastEnded = true; break }
            if step == budget - 1 { break }   // the last token is not fed back, as upstream

            for b in 0..<width {
                let embeds = try await decoder.embeddings(of: [Int32(sequences[b].last!)])
                var l = try await decoder.logits(embeddings: embeds, count: 1, sequenceLength: sequenceLength, state: beamStates[b])
                if guided {
                    let u = try await decoder.logits(embeddings: embeds, count: 1, sequenceLength: sequenceLength, state: nullStates[b])
                    l = Self.guide(conditional: l, unconditional: u, coefficient: Float(options.guidance))
                }
                logitsPerBeam[b] = l
            }
        }

        let best = scores.indices.max { scores[$0] < scores[$1] } ?? 0
        var generated = Array(sequences[best].dropFirst(1 + prompt.count))
        var ended = lastEnded
        if let end = generated.firstIndex(of: EventVocabulary.endOfChunk) {
            generated = Array(generated[..<end])
            ended = true
        }
        return Result(tokens: generated, ended: ended)
    }
}
