//
//  GRU.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Bidirectional GRU stacks with a linear head, PyTorch's semantics exactly:
//  gates ordered r, z, n; n = tanh(W_in x + b_in + r ∘ (W_hn h + b_hn));
//  h' = (1 − z) ∘ n + z ∘ h; layer L+1 reads the concatenated forward and
//  backward outputs of layer L. Core AI has no recurrent op, and unrolling the
//  piano model's sixteen recurrences over 1001 frames is a 770k-op graph, so
//  the recurrence runs here on Accelerate: one matrix product for the input
//  projection of a whole sequence, one matrix-vector product per step for the
//  hidden projection. Weights are stored pre-transposed for vDSP_mmul.
//

import Accelerate
import Foundation

/// One direction of one GRU layer.
struct GRUDirection: Sendable {
    let inputSize: Int
    let hidden: Int
    let weightIHT: [Float]  // [inputSize][3H]: PyTorch's weight_ih [3H][in], transposed
    let weightHH: [Float]   // [3H][H]
    let biasIH: [Float]     // [3H]
    let biasHH: [Float]     // [3H]

    /// Outputs `[frames][H]`, running forward or backward through `x` (`[frames][inputSize]`).
    func run(_ x: [Float], frames: Int, reverse: Bool) -> [Float] {
        let h3 = 3 * hidden
        // Input projection for every frame at once: G = X · W_ihᵀ + b_ih.
        var gates = [Float](repeating: 0, count: frames * h3)
        vDSP_mmul(x, 1, weightIHT, 1, &gates, 1, vDSP_Length(frames), vDSP_Length(h3), vDSP_Length(inputSize))
        gates.withUnsafeMutableBufferPointer { gp in
            for t in 0..<frames {
                vDSP_vadd(gp.baseAddress! + t * h3, 1, biasIH, 1, gp.baseAddress! + t * h3, 1, vDSP_Length(h3))
            }
        }

        var out = [Float](repeating: 0, count: frames * hidden)
        var h = [Float](repeating: 0, count: hidden)
        var gh = [Float](repeating: 0, count: h3)
        var r = [Float](repeating: 0, count: hidden)
        var z = [Float](repeating: 0, count: hidden)
        var n = [Float](repeating: 0, count: hidden)
        var tmp = [Float](repeating: 0, count: hidden)
        var count = Int32(hidden)
        let order: [Int] = reverse ? Array((0..<frames).reversed()) : Array(0..<frames)
        for t in order {
            // gh = W_hh · h + b_hh
            vDSP_mmul(weightHH, 1, h, 1, &gh, 1, vDSP_Length(h3), 1, vDSP_Length(hidden))
            vDSP_vadd(gh, 1, biasHH, 1, &gh, 1, vDSP_Length(h3))
            gates.withUnsafeBufferPointer { gp in
                let gi = gp.baseAddress! + t * h3
                gh.withUnsafeBufferPointer { ghp in
                    // r = σ(gi_r + gh_r), z = σ(gi_z + gh_z)
                    vDSP_vadd(gi, 1, ghp.baseAddress!, 1, &r, 1, vDSP_Length(hidden))
                    vDSP_vadd(gi + hidden, 1, ghp.baseAddress! + hidden, 1, &z, 1, vDSP_Length(hidden))
                    Self.sigmoid(&r, count: &count, scratch: &tmp)
                    Self.sigmoid(&z, count: &count, scratch: &tmp)
                    // n = tanh(gi_n + r ∘ gh_n)
                    vDSP_vmul(r, 1, ghp.baseAddress! + 2 * hidden, 1, &tmp, 1, vDSP_Length(hidden))
                    vDSP_vadd(gi + 2 * hidden, 1, tmp, 1, &n, 1, vDSP_Length(hidden))
                    vvtanhf(&n, n, &count)
                }
            }
            // h = n + z ∘ (h − n)
            vDSP_vsub(n, 1, h, 1, &tmp, 1, vDSP_Length(hidden))     // tmp = h − n
            vDSP_vmul(z, 1, tmp, 1, &tmp, 1, vDSP_Length(hidden))
            vDSP_vadd(n, 1, tmp, 1, &h, 1, vDSP_Length(hidden))
            out.replaceSubrange((t * hidden)..<((t + 1) * hidden), with: h)
        }
        return out
    }

    /// In place: x ← 1 / (1 + e^−x).
    private static func sigmoid(_ x: inout [Float], count: inout Int32, scratch: inout [Float]) {
        var negative = [Float](repeating: 0, count: x.count)
        vDSP_vneg(x, 1, &negative, 1, vDSP_Length(x.count))
        vvexpf(&scratch, negative, &count)
        var one: Float = 1
        vDSP_vsadd(scratch, 1, &one, &scratch, 1, vDSP_Length(x.count))
        vvrecf(&x, scratch, &count)
    }
}

/// A stack of bidirectional GRU layers followed by a linear layer and a sigmoid.
public struct GRUHead: Sendable {
    let layers: [(forward: GRUDirection, backward: GRUDirection)]
    let hidden: Int
    public let classes: Int
    let fcWeightT: [Float]  // [2H][classes]
    let fcBias: [Float]

    public var inputSize: Int { layers[0].forward.inputSize }

    /// The stack's output `[frames][2H]` before the head.
    func features(_ x: [Float], frames: Int) -> [Float] {
        var input = x
        for layer in layers {
            let f = layer.forward.run(input, frames: frames, reverse: false)
            let b = layer.backward.run(input, frames: frames, reverse: true)
            var joined = [Float](repeating: 0, count: frames * 2 * hidden)
            for t in 0..<frames {
                joined.replaceSubrange((t * 2 * hidden)..<(t * 2 * hidden + hidden), with: f[(t * hidden)..<((t + 1) * hidden)])
                joined.replaceSubrange((t * 2 * hidden + hidden)..<((t + 1) * 2 * hidden), with: b[(t * hidden)..<((t + 1) * hidden)])
            }
            input = joined
        }
        return input
    }

    /// `sigmoid(fc(gru(x)))`, `[frames][classes]`.
    public func run(_ x: [Float], frames: Int) -> [Float] {
        precondition(x.count == frames * inputSize)
        let feats = features(x, frames: frames)
        var out = [Float](repeating: 0, count: frames * classes)
        vDSP_mmul(feats, 1, fcWeightT, 1, &out, 1, vDSP_Length(frames), vDSP_Length(classes), vDSP_Length(2 * hidden))
        out.withUnsafeMutableBufferPointer { op in
            for t in 0..<frames {
                vDSP_vadd(op.baseAddress! + t * classes, 1, fcBias, 1, op.baseAddress! + t * classes, 1, vDSP_Length(classes))
            }
        }
        for i in out.indices { out[i] = 1 / (1 + exp(-out[i])) }
        return out
    }

    /// Read one head from a flat parameter vector in the export's order: per
    /// layer, forward then backward, `weight_ih, weight_hh, bias_ih, bias_hh`;
    /// then the head's weight and bias. Advances `cursor`.
    public static func parse(_ p: [Float], cursor: inout Int, layers layerCount: Int, inputSize: Int, hidden: Int, classes: Int) -> GRUHead {
        func take(_ n: Int) -> [Float] { defer { cursor += n }; return Array(p[cursor..<(cursor + n)]) }
        func transposed(_ m: [Float], rows: Int, columns: Int) -> [Float] {
            var t = [Float](repeating: 0, count: m.count)
            vDSP_mtrans(m, 1, &t, 1, vDSP_Length(columns), vDSP_Length(rows))
            return t
        }
        var layers: [(forward: GRUDirection, backward: GRUDirection)] = []
        for layer in 0..<layerCount {
            let inSize = layer == 0 ? inputSize : 2 * hidden
            var directions: [GRUDirection] = []
            for _ in 0..<2 {
                let wih = take(3 * hidden * inSize), whh = take(3 * hidden * hidden), bih = take(3 * hidden), bhh = take(3 * hidden)
                directions.append(GRUDirection(inputSize: inSize, hidden: hidden,
                                               weightIHT: transposed(wih, rows: 3 * hidden, columns: inSize),
                                               weightHH: whh, biasIH: bih, biasHH: bhh))
            }
            layers.append((directions[0], directions[1]))
        }
        let w = take(classes * 2 * hidden), b = take(classes)
        return GRUHead(layers: layers, hidden: hidden, classes: classes,
                       fcWeightT: transposed(w, rows: classes, columns: 2 * hidden), fcBias: b)
    }
}
