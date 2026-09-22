//
//  Sampling.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Picking the next token from a row of logits.
//

import Foundation

/// The per-step token choice, on the host, over the logits Core AI hands back.
public enum Sampling {

    /// Mask reserved rows and forbidden ids to −∞, in place.
    ///
    /// Rows at or past ``EventVocabulary/count`` exist only in the wider
    /// checkpoints and were never trained; upstream masks them every step.
    public static func mask(_ logits: inout [Float], forbidden: [Int]) {
        if logits.count > EventVocabulary.count {
            for i in EventVocabulary.count..<logits.count { logits[i] = -.infinity }
        }
        for id in forbidden where id < logits.count { logits[id] = -.infinity }
    }

    /// The first index of the largest value — the tie-break `torch.argmax` uses.
    public static func argmax(_ logits: [Float]) -> Int {
        var best = 0
        for i in 1..<logits.count where logits[i] > logits[best] { best = i }
        return best
    }

    /// Draw from `softmax(logits / temperature)`.
    ///
    /// A temperature of zero falls back to the argmax, as upstream's
    /// `use_sampling and temp > 0.0` does.
    public static func sample<G: RandomNumberGenerator>(_ logits: [Float], temperature: Double, using generator: inout G) -> Int {
        guard temperature > 0 else { return argmax(logits) }
        let peak = logits.max() ?? 0
        var probabilities = logits.map { $0 == -.infinity ? 0 : exp(Double($0 - peak) / temperature) }
        let total = probabilities.reduce(0, +)
        guard total > 0 else { return argmax(logits) }
        for i in probabilities.indices { probabilities[i] /= total }
        var u = Double.random(in: 0..<1, using: &generator)
        for (i, p) in probabilities.enumerated() {
            u -= p
            if u < 0 { return i }
        }
        return probabilities.lastIndex { $0 > 0 } ?? argmax(logits)
    }

    /// `log_softmax` of a row, for beam scoring.
    public static func logSoftmax(_ logits: [Float]) -> [Double] {
        let peak = Double(logits.max() ?? 0)
        let exps = logits.map { $0 == -.infinity ? 0.0 : exp(Double($0) - peak) }
        let logTotal = log(exps.reduce(0, +)) + peak
        return logits.map { $0 == -.infinity ? -.infinity : Double($0) - logTotal }
    }
}

/// A seedable generator so `sampling` runs can be reproduced. SplitMix64.
public struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
