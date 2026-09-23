//
//  PianoFrontEnd.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The log-mel front end of the piano transcription model (Kong et al. 2020):
//  torchlibrosa's Spectrogram + LogmelFilterBank as the checkpoint configures
//  them — 16 kHz, n_fft 2048, hop 160, periodic Hann, reflect-centred, power
//  spectrum, 229 Slaney-scale mel bands from 30 Hz to 8 kHz with librosa's
//  Slaney area normalisation, then 10·log10(max(1e-10, x)). Ten-second
//  segments in, 1001 frames of 229 out. The model's own bn0 sits in the graph.
//

import Accelerate
import Foundation

/// Log-mel frames for one ten-second segment, as the piano model was trained on.
public struct PianoFrontEnd: Sendable {

    public static let sampleRate = 16_000
    public static let mels = 229
    public static let framesPerSecond = 100
    /// One segment: ten seconds.
    public static let segmentSamples = 160_000
    /// Frames per segment with centre padding: 1 + segmentSamples / hop.
    public static let segmentFrames = 1001
    static let nFFT = 2048
    static let hop = 160
    static let bins = nFFT / 2 + 1   // 1025
    static let fMin = 30.0
    static let fMax = 8000.0
    static let floor: Float = 1e-10

    private let filterbank: [Float]   // [bins][mels], row-major
    private let window: [Float]

    public init() {
        filterbank = Self.slaneyFilterbank()
        // librosa.filters.get_window("hann", 2048, fftbins=True): periodic.
        window = (0..<Self.nFFT).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(Self.nFFT))) }
    }

    /// `[segmentFrames][mels]` for exactly ``segmentSamples`` samples.
    public func logMel(segment: [Float]) -> [Float] {
        precondition(segment.count == Self.segmentSamples, "a segment is \(Self.segmentSamples) samples")
        let half = Self.nFFT / 2
        var padded = [Float]()
        padded.reserveCapacity(segment.count + Self.nFFT)
        padded.append(contentsOf: (1...half).reversed().map { segment[$0] })
        padded.append(contentsOf: segment)
        padded.append(contentsOf: (2...(half + 1)).map { segment[segment.count - $0] })

        guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD) else {
            preconditionFailure("could not create a \(Self.nFFT)-point DFT")
        }
        defer { vDSP_DFT_DestroySetup(setup) }

        var inRe = [Float](repeating: 0, count: Self.nFFT)
        let inIm = [Float](repeating: 0, count: Self.nFFT)
        var outRe = [Float](repeating: 0, count: Self.nFFT)
        var outIm = [Float](repeating: 0, count: Self.nFFT)
        var power = [Float](repeating: 0, count: Self.segmentFrames * Self.bins)
        for f in 0..<Self.segmentFrames {
            let start = f * Self.hop
            vDSP_vmul(Array(padded[start..<(start + Self.nFFT)]), 1, window, 1, &inRe, 1, vDSP_Length(Self.nFFT))
            vDSP_DFT_Execute(setup, inRe, inIm, &outRe, &outIm)
            for b in 0..<Self.bins {
                power[f * Self.bins + b] = outRe[b] * outRe[b] + outIm[b] * outIm[b]
            }
        }

        var mel = [Float](repeating: 0, count: Self.segmentFrames * Self.mels)
        vDSP_mmul(power, 1, filterbank, 1, &mel, 1,
                  vDSP_Length(Self.segmentFrames), vDSP_Length(Self.mels), vDSP_Length(Self.bins))
        for i in mel.indices { mel[i] = 10 * log10(max(Self.floor, mel[i])) }
        return mel
    }

    // MARK: - Filterbank

    /// `librosa.filters.mel(sr=16000, n_fft=2048, n_mels=229, fmin=30, fmax=8000)`:
    /// Slaney scale (linear below 1 kHz, logarithmic above) with `norm="slaney"`,
    /// each triangle scaled by 2 / (its bandwidth), as `[bins][mels]`.
    static func slaneyFilterbank() -> [Float] {
        let nFreqs = bins, nMels = mels
        let allFreqs = (0..<nFreqs).map { Double($0) * Double(sampleRate) / 2 / Double(nFreqs - 1) }
        let fSp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / fSp, logstep = log(6.4) / 27
        func hzToMel(_ f: Double) -> Double { f >= minLogHz ? minLogMel + log(f / minLogHz) / logstep : f / fSp }
        func melToHz(_ m: Double) -> Double { m >= minLogMel ? minLogHz * exp(logstep * (m - minLogMel)) : m * fSp }
        let mMin = hzToMel(fMin), mMax = hzToMel(fMax)
        let fPts = (0..<(nMels + 2)).map { melToHz(mMin + (mMax - mMin) * Double($0) / Double(nMels + 1)) }
        let fDiff = (0..<(nMels + 1)).map { fPts[$0 + 1] - fPts[$0] }
        var fb = [Float](repeating: 0, count: nFreqs * nMels)
        for m in 0..<nMels {
            let enorm = 2 / (fPts[m + 2] - fPts[m])
            for i in 0..<nFreqs {
                let lower = (allFreqs[i] - fPts[m]) / fDiff[m]
                let upper = (fPts[m + 2] - allFreqs[i]) / fDiff[m + 1]
                fb[i * nMels + m] = Float(max(0, min(lower, upper)) * enorm)
            }
        }
        return fb
    }
}
