//
//  Auralizer.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  A check mix: the original on the left, the transcription synthesised on the
//  right, so a listener can hear what the model got. Upstream renders with
//  FluidSynth; macOS has a General MIDI synthesiser built in, so this needs no
//  external program, and takes the same SoundFont if one is given.
//

import AVFAudio
import AudioToolbox
import Foundation
import MIDIFileKit

/// Renders MIDI to audio on this machine and mixes it against the original.
public enum Auralizer {

    /// The render rate, upstream's.
    public static let sampleRate = 44_100.0

    /// Write a stereo WAV: left the original, right the synthesised MIDI,
    /// RMS-matched to the original's loudness.
    ///
    /// - Parameters:
    ///   - midi: the MIDI file's bytes. Its bar-offset marker, if any, is undone
    ///     so the synthesis lines up with the original.
    ///   - original: the recording that was transcribed.
    ///   - destination: the WAV to write.
    ///   - soundfont: an `.sf2` to render with; nil uses the system's GM bank.
    public static func render(midi: Data, original: URL, to destination: URL, soundfont: URL? = nil) throws {
        var synth = try synthesize(midi: midi, soundfont: soundfont)
        let file = try MIDIReader.read(midi)
        if let marker = file.markers.first(where: { $0.text.hasPrefix(BeatGridMath.barOffsetMarker) }),
           let offset = Double(marker.text.dropFirst(BeatGridMath.barOffsetMarker.count)), offset > 0 {
            let skip = min(synth.count, Int((offset * sampleRate).rounded()))
            synth.removeFirst(skip)
        }
        var source = try AudioLoader.load(original, sampleRate: Int(sampleRate))
        let length = max(source.count, synth.count)
        source += [Float](repeating: 0, count: length - source.count)
        synth += [Float](repeating: 0, count: length - synth.count)
        let rmsSource = rms(source), rmsSynth = rms(synth)
        if rmsSynth > 1e-8 {
            let gain = rmsSource / rmsSynth
            for i in synth.indices { synth[i] *= gain }
        }
        try AudioWriter.writeWAV(channels: [source, synth], sampleRate: sampleRate, to: destination)
    }

    /// Render `midi` to mono samples at ``sampleRate`` with the built-in
    /// General MIDI synthesiser, offline.
    public static func synthesize(midi: Data, soundfont: URL? = nil) throws -> [Float] {
        let engine = AVAudioEngine()
        let description = AudioComponentDescription(componentType: kAudioUnitType_MusicDevice,
                                                    componentSubType: kAudioUnitSubType_DLSSynth,
                                                    componentManufacturer: kAudioUnitManufacturer_Apple,
                                                    componentFlags: 0, componentFlagsMask: 0)
        let synth = AVAudioUnitMIDIInstrument(audioComponentDescription: description)
        engine.attach(synth)
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        engine.connect(synth, to: engine.mainMixerNode, format: format)
        engine.connect(engine.mainMixerNode, to: engine.outputNode, format: format)
        if let soundfont {
            var url = soundfont as CFURL
            let status = AudioUnitSetProperty(synth.audioUnit, kMusicDeviceProperty_SoundBankURL,
                                              kAudioUnitScope_Global, 0, &url, UInt32(MemoryLayout<CFURL>.size))
            guard status == noErr else {
                throw MusicTranscriberError.inferenceFailed("the synthesiser refused \(soundfont.lastPathComponent) (status \(status))")
            }
        }

        let sequencer = AVAudioSequencer(audioEngine: engine)
        do { try sequencer.load(from: midi, options: []) } catch {
            throw MusicTranscriberError.inferenceFailed("could not load the MIDI into the sequencer: \(error.localizedDescription)")
        }
        for track in sequencer.tracks { track.destinationAudioUnit = synth }
        // The length from the file's own tempo map — the sequencer's estimate ran
        // long — plus two seconds of release tail after the last note ends.
        let seconds: Double
        do { seconds = try MIDIReader.read(midi).duration } catch {
            throw MusicTranscriberError.inferenceFailed("could not read the MIDI to size the render: \(error.localizedDescription)")
        }
        let totalFrames = Int((seconds + 2) * sampleRate)

        let blockSize: AVAudioFrameCount = 4096
        do {
            try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: blockSize)
            try engine.start()
            sequencer.prepareToPlay()
            try sequencer.start()
        } catch {
            throw MusicTranscriberError.inferenceFailed("offline rendering failed to start: \(error.localizedDescription)")
        }
        defer { sequencer.stop(); engine.stop() }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: blockSize) else {
            throw MusicTranscriberError.inferenceFailed("could not allocate a render buffer")
        }
        var mono = [Float]()
        mono.reserveCapacity(totalFrames)
        while mono.count < totalFrames {
            let frames = AVAudioFrameCount(min(Int(blockSize), totalFrames - mono.count))
            let status = try engine.renderOffline(frames, to: buffer)
            guard status == .success, let data = buffer.floatChannelData else {
                throw MusicTranscriberError.inferenceFailed("offline render returned \(status.rawValue)")
            }
            let count = Int(buffer.frameLength)
            let channels = Int(buffer.format.channelCount)
            for i in 0..<count {
                var sum: Float = 0
                for c in 0..<channels { sum += data[c][i] }
                mono.append(sum / Float(channels))
            }
        }
        return mono
    }

    static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Double = 0
        for s in samples { sum += Double(s * s) }
        return Float((sum / Double(samples.count)).squareRoot())
    }
}

/// Writing PCM out.
public enum AudioWriter {

    /// Write channels of floats as a 16-bit PCM WAV.
    public static func writeWAV(channels: [[Float]], sampleRate: Double, to url: URL) throws {
        let count = channels.map(\.count).max() ?? 0
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels.count,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        try? FileManager.default.removeItem(at: url)
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(max(count, 1))) else {
            throw MusicTranscriberError.inferenceFailed("could not allocate a write buffer")
        }
        for (c, samples) in channels.enumerated() {
            let channel = buffer.floatChannelData![c]
            for i in 0..<count { channel[i] = i < samples.count ? max(-1, min(1, samples[i])) : 0 }
        }
        buffer.frameLength = AVAudioFrameCount(count)
        try file.write(from: buffer)
    }
}
