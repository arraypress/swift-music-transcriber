//
//  AudioLoader.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Any decodable file to the model's 16 kHz mono float stream.
//

import AVFAudio
import Foundation

/// Reads audio the way upstream's `load_audio` does: decode, average the
/// channels to mono, then sinc-resample to 16 kHz.
public enum AudioLoader {

    /// Mono samples at `sampleRate`, the model's 16 kHz by default.
    public static func load(_ url: URL, sampleRate target: Int = MelSpectrogram.sampleRate) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        } catch {
            throw MusicTranscriberError.audioUnreadable("\(url.lastPathComponent): \(error.localizedDescription)")
        }
        let format = file.processingFormat
        let channelCount = Int(format.channelCount)
        // Read to end-of-file rather than to `file.length`: for a float WAV written
        // by libsndfile that property came back 611 frames short of the data chunk,
        // which silently lost the last chunk of a transcription.
        let capacity = AVAudioFrameCount(65_536)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
            throw MusicTranscriberError.audioUnreadable("\(url.lastPathComponent): cannot allocate a buffer")
        }
        var mono: [Float] = []
        mono.reserveCapacity(Int(max(file.length, 0)))
        while true {
            do { try file.read(into: buffer, frameCount: capacity) } catch let error as NSError {
                // kAudioFileEndOfFileError: a read at the true end of the data, which
                // the short `length` never announced. Everything before it is kept.
                if error.code == -39 { break }
                throw MusicTranscriberError.audioUnreadable("\(url.lastPathComponent): \(error.localizedDescription)")
            }
            let count = Int(buffer.frameLength)
            if count == 0 { break }
            guard let channels = buffer.floatChannelData else {
                throw MusicTranscriberError.audioUnreadable("\(url.lastPathComponent): not float PCM")
            }
            let base = mono.count
            mono.append(contentsOf: [Float](repeating: 0, count: count))
            for c in 0..<channelCount {
                let channel = channels[c]
                for i in 0..<count { mono[base + i] += channel[i] }
            }
        }
        guard !mono.isEmpty else {
            throw MusicTranscriberError.audioUnreadable("\(url.lastPathComponent) is empty")
        }
        if channelCount > 1 {
            let scale = 1 / Float(channelCount)
            for i in mono.indices { mono[i] *= scale }
        }
        let rate = Int(format.sampleRate.rounded())
        guard rate != target else { return mono }
        return Resampler(from: rate, to: target).resample(mono)
    }

    /// Split 16 kHz mono samples into 5-second chunks; the last is short and the
    /// front end pads it.
    public static func chunks(_ samples: [Float]) -> [ArraySlice<Float>] {
        stride(from: 0, to: max(samples.count, 1), by: MelSpectrogram.chunkSamples).map {
            samples[$0..<min($0 + MelSpectrogram.chunkSamples, samples.count)]
        }
    }
}
