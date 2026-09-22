//
//  MelSpectrogram.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The model's audio front end, reproduced bit-for-bit from torchaudio.
//
//  The conditioner upstream is `torchaudio.transforms.MelSpectrogram` with
//  n_fft 2048, hop 160, 512 HTK mel bins, no normalisation, magnitude (not
//  power), centre-padded by reflection, then log(x + 1e-6). Every one of those
//  choices is visible in the output and the model was trained on exactly this,
//  so this file makes them all explicit and the tests hold it to a fixture
//  dumped from the Python transform.
//

import Accelerate
import Foundation

/// Log-mel frames for one 5-second chunk.
public struct MelSpectrogram: Sendable {

    /// The model's sample rate.
    public static let sampleRate = 16_000

    /// Samples per chunk: 5 seconds.
    public static let chunkSamples = 80_000

    /// Frames per chunk with centre padding: 1 + 80000 / 160.
    public static let frames = 501

    public static let mels = 512
    static let nFFT = 2048
    static let hop = 160
    static let bins = nFFT / 2 + 1   // 1025

    private let filterbank: [Float]   // [bins][mels], row-major
    private let window: [Float]

    public init() {
        filterbank = Self.htkFilterbank()
        // torch.hann_window is periodic: 0.5 − 0.5·cos(2πn/N).
        window = (0..<Self.nFFT).map { 0.5 * (1 - cos(2 * Float.pi * Float($0) / Float(Self.nFFT))) }
    }

    /// Compute `[frames][mels]` log-mel values for exactly ``chunkSamples`` samples.
    ///
    /// Shorter input is zero-padded to a chunk first, which is what upstream
    /// does to the last chunk of a file.
    public func logMel(_ samples: [Float]) -> [Float] {
        precondition(samples.count <= Self.chunkSamples, "a chunk is at most \(Self.chunkSamples) samples")
        var wav = samples
        if wav.count < Self.chunkSamples {
            wav.append(contentsOf: [Float](repeating: 0, count: Self.chunkSamples - wav.count))
        }

        // Reflect padding of n_fft/2 on both sides, as torch.stft(center=True).
        let half = Self.nFFT / 2
        var padded = [Float]()
        padded.reserveCapacity(wav.count + Self.nFFT)
        padded.append(contentsOf: (1...half).reversed().map { wav[$0] })
        padded.append(contentsOf: wav)
        padded.append(contentsOf: (2...(half + 1)).map { wav[wav.count - $0] })

        guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD) else {
            preconditionFailure("could not create a \(Self.nFFT)-point DFT")
        }
        defer { vDSP_DFT_DestroySetup(setup) }

        var inRe = [Float](repeating: 0, count: Self.nFFT)
        let inIm = [Float](repeating: 0, count: Self.nFFT)
        var outRe = [Float](repeating: 0, count: Self.nFFT)
        var outIm = [Float](repeating: 0, count: Self.nFFT)
        var magnitude = [Float](repeating: 0, count: Self.frames * Self.bins)

        for f in 0..<Self.frames {
            let start = f * Self.hop
            vDSP_vmul(Array(padded[start..<(start + Self.nFFT)]), 1, window, 1, &inRe, 1, vDSP_Length(Self.nFFT))
            vDSP_DFT_Execute(setup, inRe, inIm, &outRe, &outIm)
            for b in 0..<Self.bins {
                magnitude[f * Self.bins + b] = (outRe[b] * outRe[b] + outIm[b] * outIm[b]).squareRoot()
            }
        }

        var mel = [Float](repeating: 0, count: Self.frames * Self.mels)
        vDSP_mmul(magnitude, 1, filterbank, 1, &mel, 1,
                  vDSP_Length(Self.frames), vDSP_Length(Self.mels), vDSP_Length(Self.bins))
        for i in mel.indices { mel[i] = log(mel[i] + 1e-6) }
        return mel
    }

    // MARK: - Filterbank

    /// `torchaudio.functional.melscale_fbanks(1025, 0, 8000, 512, 16000)` with
    /// `mel_scale="htk"` and no norm: triangular filters at equal spacing on the
    /// HTK mel scale, returned as `[bins][mels]`.
    static func htkFilterbank() -> [Float] {
        let nFreqs = bins, nMels = mels
        let allFreqs = (0..<nFreqs).map { Double($0) * Double(sampleRate / 2) / Double(nFreqs - 1) }
        func hzToMel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        func melToHz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let mMin = hzToMel(0), mMax = hzToMel(Double(sampleRate) / 2)
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
