//
//  TranscriptionOptions.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Every knob the upstream `transcribe()` exposes, with the same defaults.
//

import Foundation

/// How to decode.
///
/// The defaults are upstream's: greedy, no guidance, prelude forcing on, every
/// instrument allowed. Each field documents what changing it trades away.
public struct TranscriptionOptions: Sendable, Codable, Equatable {

    /// Sample from the softmax instead of taking the argmax. Non-deterministic
    /// unless ``seed`` is set. Off by default because the argmax is what the
    /// model was evaluated with.
    public var sampling = false

    /// Softmax temperature, used only with ``sampling``. 1.0 is the raw distribution.
    public var temperature = 1.0

    /// Classifier-free guidance strength. 1.0 is off. Anything else costs a
    /// second forward pass per token, against a null condition, and upstream
    /// marks it "todo: make it dynamic" — treat it as experimental.
    public var guidance = 1.0

    /// Restrict the output to these groups. A hard constraint: every other
    /// program and drum token is masked to −∞ at each step, and the groups are
    /// also passed to the model as conditioning. Empty means anything goes.
    public var instruments: [InstrumentGroup] = []

    /// Teacher-force each chunk's tie prologue from the previous chunk's
    /// still-sounding notes, so a chunk cannot restart them on the wrong
    /// instrument. Requires chunks in order; on by default.
    public var preludeForcing = true

    /// Chunks decoded per pass. Upstream lets a GPU batch several chunks at the
    /// cost of prelude forcing. This engine decodes one chunk at a time on
    /// Metal whatever the value; the *semantics* — forcing off above 1 — are
    /// kept so the output matches upstream's for the same settings.
    public var batchSize = 1

    /// Throw when a chunk fails to emit its end token within the budget instead
    /// of warning. Off by default: the budget runs out on dense material and
    /// the notes decoded so far are still worth having.
    public var strictEndOfChunk = false

    /// Beam width. 1 is greedy or sampling; ≥ 2 runs a length-normalised beam
    /// search, which is not streamed and costs one cache copy per beam per step.
    public var beamSize = 1

    /// The generation budget per chunk, in tokens, prompt included. 2,000 is
    /// upstream's constant; a 5-second chunk of dense electronic music can hit it.
    public var maximumTokens = 2000

    /// Seed for ``sampling``. Nil draws from the system generator.
    public var seed: UInt64?

    public init() {}

    /// Upstream refuses this combination rather than quietly dropping the
    /// forcing, and so does this.
    public func validate() throws {
        if preludeForcing && batchSize > 1 {
            throw MusicTranscriberError.invalidOptions(
                "batchSize \(batchSize) disables prelude forcing, which lowers quality at chunk boundaries; set preludeForcing = false to accept that")
        }
        if beamSize < 1 { throw MusicTranscriberError.invalidOptions("beamSize must be at least 1") }
        if temperature < 0 { throw MusicTranscriberError.invalidOptions("temperature cannot be negative") }
    }
}
