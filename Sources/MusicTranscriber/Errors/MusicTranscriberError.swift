//
//  MusicTranscriberError.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// Everything this library can fail with.
public enum MusicTranscriberError: Error, LocalizedError, Sendable, Equatable {

    /// No model asset where one was expected.
    case modelNotFound(String)

    /// The asset loaded but is not a MuScriptor export this library understands.
    case modelIncompatible(String)

    /// The audio file could not be read or decoded.
    case audioUnreadable(String)

    /// An instrument name matched nothing.
    case unknownInstrument(String, suggestions: [String])

    /// An instrument name matched several groups.
    case ambiguousInstrument(String, candidates: [String])

    /// Options that upstream would refuse.
    case invalidOptions(String)

    /// A chunk ran out of budget before its end token, under strict mode.
    case chunkDidNotEnd(chunk: Int, seekTime: Double, budget: Int)

    /// The beat tracker found no usable grid, under ``TempoDetection/required``.
    case noSteadyTempo(String)

    /// Core AI refused to run.
    case inferenceFailed(String)

    /// An external program this feature shells out to is missing.
    case externalToolMissing(String, hint: String)

    /// A feature that is not built yet.
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let location): return "no model at \(location)"
        case .modelIncompatible(let why): return "incompatible model: \(why)"
        case .audioUnreadable(let why): return "cannot read audio: \(why)"
        case .unknownInstrument(let name, let suggestions):
            return "unknown instrument name '\(name)'"
                + (suggestions.isEmpty ? "" : " — did you mean \(suggestions.joined(separator: ", "))?")
        case .ambiguousInstrument(let name, let candidates):
            return "ambiguous instrument name '\(name)': matches \(candidates.joined(separator: ", "))"
        case .invalidOptions(let why): return why
        case .chunkDidNotEnd(let chunk, let seek, let budget):
            return "chunk \(chunk) (seek=\(String(format: "%.1f", seek))s) did not emit its end token within \(budget) tokens"
        case .noSteadyTempo(let why): return why
        case .inferenceFailed(let why): return "inference failed: \(why)"
        case .externalToolMissing(let tool, let hint): return "\(tool) is not installed. \(hint)"
        case .unsupported(let what): return "\(what) is not supported yet"
        }
    }
}
