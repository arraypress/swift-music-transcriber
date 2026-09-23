//
//  BeatThisFrontEnd.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Beat This!'s spectrogram, reproduced from torchaudio: 22.05 kHz, 1024-point
//  frames every 441 samples (50 per second), magnitude divided by √1024,
//  128 Slaney-scale mel bands from 30 Hz to 11 kHz, then log(1 + 1000·x).
//
//  MuScriptor hands the tracker its own 16 kHz signal and the tracker
//  resamples it to 22.05 kHz with soxr; here the fleet's julius port does that
//  step, which is the one place the two front ends are not the same code.
//

import Accelerate
import Foundation

/// Log-mel frames for the beat tracker.
public struct BeatThisFrontEnd: Sendable {

    public static let sampleRate = 22_050
    public static let mels = 128
    public static let framesPerSecond = 50
    static let nFFT = 1024
    static let hop = 441
    static let bins = nFFT / 2 + 1   // 513

    private let filterbank: [Float]   // [bins][mels]
    private let window: [Float]

    public init() {
        filterbank = Self.slaneyFilterbank()
        window = (0..<Self.nFFT).map { 0.5 * (1 - cos(2 * Float.pi * Float($0) / Float(Self.nFFT))) }
    }

    /// Frames for a 22.05 kHz mono signal: `1 + count / 441` rows of 128.
    public func logMel(_ samples: [Float]) -> (values: [Float], frames: Int) {
        let frames = 1 + samples.count / Self.hop
        let half = Self.nFFT / 2
        var padded = [Float]()
        padded.reserveCapacity(samples.count + Self.nFFT)
        // torch.stft(center=True) reflects; a signal shorter than the pad is
        // reflected as far as it goes, which never happens at these lengths.
        padded.append(contentsOf: (1...half).reversed().map { samples[min($0, samples.count - 1)] })
        padded.append(contentsOf: samples)
        padded.append(contentsOf: (2...(half + 1)).map { samples[max(samples.count - $0, 0)] })

        guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD) else {
            preconditionFailure("could not create a \(Self.nFFT)-point DFT")
        }
        defer { vDSP_DFT_DestroySetup(setup) }
        var inRe = [Float](repeating: 0, count: Self.nFFT)
        let inIm = [Float](repeating: 0, count: Self.nFFT)
        var outRe = [Float](repeating: 0, count: Self.nFFT)
        var outIm = [Float](repeating: 0, count: Self.nFFT)
        var magnitude = [Float](repeating: 0, count: frames * Self.bins)
        let norm = 1 / Float(Self.nFFT).squareRoot()
        for f in 0..<frames {
            let start = f * Self.hop
            vDSP_vmul(Array(padded[start..<(start + Self.nFFT)]), 1, window, 1, &inRe, 1, vDSP_Length(Self.nFFT))
            vDSP_DFT_Execute(setup, inRe, inIm, &outRe, &outIm)
            for b in 0..<Self.bins {
                magnitude[f * Self.bins + b] = (outRe[b] * outRe[b] + outIm[b] * outIm[b]).squareRoot() * norm
            }
        }
        var mel = [Float](repeating: 0, count: frames * Self.mels)
        vDSP_mmul(magnitude, 1, filterbank, 1, &mel, 1, vDSP_Length(frames), vDSP_Length(Self.mels), vDSP_Length(Self.bins))
        for i in mel.indices { mel[i] = log1p(1000 * mel[i]) }
        return (mel, frames)
    }

    /// `torchaudio.functional.melscale_fbanks(513, 30, 11000, 128, 22050, mel_scale="slaney")`,
    /// no norm: linear below 1 kHz, logarithmic above, as `[bins][mels]`.
    static func slaneyFilterbank() -> [Float] {
        let nFreqs = bins, nMels = mels
        let allFreqs = (0..<nFreqs).map { Double($0) * Double(sampleRate / 2) / Double(nFreqs - 1) }
        let minLogHz = 1000.0, minLogMel = (minLogHz - 0) / (200.0 / 3), logstep = log(6.4) / 27
        func hzToMel(_ f: Double) -> Double { f >= minLogHz ? minLogMel + log(f / minLogHz) / logstep : f / (200.0 / 3) }
        func melToHz(_ m: Double) -> Double { m >= minLogMel ? minLogHz * exp(logstep * (m - minLogMel)) : m * (200.0 / 3) }
        let mMin = hzToMel(30), mMax = hzToMel(11_000)
        let fPts = (0..<(nMels + 2)).map { melToHz(mMin + (mMax - mMin) * Double($0) / Double(nMels + 1)) }
        let fDiff = (0..<(nMels + 1)).map { fPts[$0 + 1] - fPts[$0] }
        var fb = [Float](repeating: 0, count: nFreqs * nMels)
        for i in 0..<nFreqs {
            for m in 0..<nMels {
                let down = -(fPts[m] - allFreqs[i]) / fDiff[m]
                let up = (fPts[m + 2] - allFreqs[i]) / fDiff[m + 1]
                fb[i * nMels + m] = Float(max(0, min(down, up)))
            }
        }
        return fb
    }
}
