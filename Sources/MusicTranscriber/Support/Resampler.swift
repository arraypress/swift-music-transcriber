//
//  Resampler.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Julius' sinc resampler, so any input rate reaches the model through the same
//  filter upstream uses — AVAudioConverter would be close, and "close" moves
//  mel bins enough to change tokens on a borderline note.
//

import Accelerate
import Foundation

/// Windowed-sinc resampling between integer rates, as `julius.resample_frac`.
///
/// After dividing both rates by their GCD, one FIR kernel is built per output
/// phase (`new` of them) and applied with a stride of `old`. 44.1 kHz to 16 kHz
/// reduces to 441 : 160 — 160 kernels of 581 taps; 48 kHz to 16 kHz is 3 : 1.
public struct Resampler: Sendable {

    public let from: Int
    public let to: Int
    private let old: Int
    private let new: Int
    private let width: Int
    private let kernels: [[Float]]

    /// - Parameters:
    ///   - zeros: zero crossings kept on each side of the sinc; 24 upstream.
    ///   - rolloff: the low-pass sits at `rolloff · min(rate) / 2`; 0.945 upstream.
    public init(from: Int, to: Int, zeros: Int = 24, rolloff: Double = 0.945) {
        self.from = from
        self.to = to
        let g = gcd(from, to)
        old = from / g
        new = to / g
        guard old != new else {
            width = 0
            kernels = []
            return
        }
        let sr = Double(min(new, old)) * rolloff
        width = Int((Double(zeros) * Double(old) / sr).rounded(.up))
        let taps = 2 * width + old
        var built: [[Float]] = []
        for i in 0..<new {
            var kernel = [Float](repeating: 0, count: taps)
            var sum = 0.0
            for k in 0..<taps {
                let idx = Double(k - width)
                var t = (-Double(i) / Double(new) + idx / Double(old)) * sr
                t = min(max(t, -Double(zeros)), Double(zeros))
                t *= Double.pi
                let window = pow(cos(t / Double(zeros) / 2), 2)
                let sinc = t == 0 ? 1.0 : sin(t) / t
                kernel[k] = Float(sinc * window)
                sum += Double(kernel[k])
            }
            for k in 0..<taps { kernel[k] = Float(Double(kernel[k]) / sum) }
            built.append(kernel)
        }
        kernels = built
    }

    /// Resample `x`. Output length is `floor(to · count / from)`.
    public func resample(_ x: [Float]) -> [Float] {
        guard old != new, !x.isEmpty else { return x }
        let taps = 2 * width + old
        // Replicate padding: `width` on the left, `width + old` on the right.
        var padded = [Float](repeating: x[0], count: width)
        padded.append(contentsOf: x)
        padded.append(contentsOf: [Float](repeating: x[x.count - 1], count: width + old))
        let positions = (padded.count - taps) / old + 1
        let outputLength = Int(Double(new) * Double(x.count) / Double(old))
        var y = [Float](repeating: 0, count: positions * new)
        padded.withUnsafeBufferPointer { p in
            for i in 0..<new {
                kernels[i].withUnsafeBufferPointer { k in
                    for j in 0..<positions {
                        var dot: Float = 0
                        vDSP_dotpr(p.baseAddress! + j * old, 1, k.baseAddress!, 1, &dot, vDSP_Length(taps))
                        y[j * new + i] = dot
                    }
                }
            }
        }
        return Array(y.prefix(outputLength))
    }
}

private func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
