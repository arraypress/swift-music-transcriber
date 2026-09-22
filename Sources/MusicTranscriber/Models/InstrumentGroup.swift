//
//  InstrumentGroup.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The 36 instrument groups the model knows, and how they map to General MIDI.
//

import Foundation

/// One of the model's instrument classes.
///
/// MuScriptor does not predict a General MIDI program; it predicts one of 35
/// pitched groups plus drums (the `MT3_FULL_PLUS` taxonomy). Each pitched group
/// covers several GM programs and the model always emits the group's *first*
/// program, so a transcription of a Rhodes comes back as program 2, not 4.
/// ``programs`` is the whole set for callers that want to render differently.
public struct InstrumentGroup: Hashable, Codable, Sendable, Identifiable {

    /// The upstream name, e.g. `"clean_electric_guitar"`. Stable: it is the
    /// spelling the CLI's `--instruments` accepts and the JSON output uses.
    public let name: String

    /// The conditioning class index the model was trained with. Never renumber.
    public let id: Int

    /// General MIDI programs this group stands for, representative first.
    /// Empty for drums, which live on channel 10 rather than a program.
    public let programs: [Int]

    /// Whether this is the drum kit.
    public var isDrums: Bool { programs.isEmpty }

    /// The GM program the model emits for this group, or 128 for drums —
    /// the sentinel upstream uses for "not a program".
    public var representativeProgram: Int { programs.first ?? InstrumentGroup.drumProgram }

    /// The name with spaces, for track names and tables.
    public var displayName: String { name.replacingOccurrences(of: "_", with: " ") }

    /// Upstream's pseudo-program for drums.
    public static let drumProgram = 128

    // MARK: - The table

    /// Every group, in class-index order. Drums are index 36; 34 and 35 are the two
    /// sound-effect groups, which the published names table skips.
    public static let all: [InstrumentGroup] = [
        InstrumentGroup(name: "acoustic_piano", id: 0, programs: [0, 1, 3, 6, 7]),
        InstrumentGroup(name: "electric_piano", id: 1, programs: [2, 4, 5]),
        InstrumentGroup(name: "chromatic_percussion", id: 2, programs: Array(8...15)),
        InstrumentGroup(name: "organ", id: 3, programs: Array(16...23)),
        InstrumentGroup(name: "acoustic_guitar", id: 4, programs: [24, 25]),
        InstrumentGroup(name: "clean_electric_guitar", id: 5, programs: [26, 27, 28]),
        InstrumentGroup(name: "distorted_electric_guitar", id: 6, programs: [29, 30, 31]),
        InstrumentGroup(name: "acoustic_bass", id: 7, programs: [32, 35]),
        InstrumentGroup(name: "electric_bass", id: 8, programs: [33, 34, 36, 37, 38, 39]),
        InstrumentGroup(name: "violin", id: 9, programs: [40]),
        InstrumentGroup(name: "viola", id: 10, programs: [41]),
        InstrumentGroup(name: "cello", id: 11, programs: [42]),
        InstrumentGroup(name: "contrabass", id: 12, programs: [43]),
        InstrumentGroup(name: "orchestral_harp", id: 13, programs: [46]),
        InstrumentGroup(name: "timpani", id: 14, programs: [47]),
        InstrumentGroup(name: "string_ensemble", id: 15, programs: [48, 49, 44, 45]),
        InstrumentGroup(name: "synth_strings", id: 16, programs: [50, 51]),
        InstrumentGroup(name: "voice", id: 17, programs: [52, 53, 54]),
        InstrumentGroup(name: "orchestra_hit", id: 18, programs: [55]),
        InstrumentGroup(name: "trumpet", id: 19, programs: [56, 59]),
        InstrumentGroup(name: "trombone", id: 20, programs: [57]),
        InstrumentGroup(name: "tuba", id: 21, programs: [58]),
        InstrumentGroup(name: "french_horn", id: 22, programs: [60]),
        InstrumentGroup(name: "brass_section", id: 23, programs: [61, 62, 63]),
        InstrumentGroup(name: "soprano_and_alto_sax", id: 24, programs: [64, 65]),
        InstrumentGroup(name: "tenor_sax", id: 25, programs: [66]),
        InstrumentGroup(name: "baritone_sax", id: 26, programs: [67]),
        InstrumentGroup(name: "oboe", id: 27, programs: [68]),
        InstrumentGroup(name: "english_horn", id: 28, programs: [69]),
        InstrumentGroup(name: "bassoon", id: 29, programs: [70]),
        InstrumentGroup(name: "clarinet", id: 30, programs: [71]),
        InstrumentGroup(name: "flutes", id: 31, programs: Array(72...79)),
        InstrumentGroup(name: "synth_lead", id: 32, programs: Array(80...87)),
        InstrumentGroup(name: "synth_pad", id: 33, programs: Array(88...95)),
        InstrumentGroup(name: "drums", id: 36, programs: []),
    ]

    /// The names `--instruments` accepts, in table order.
    public static var names: [String] { all.map(\.name) }

    /// Look a group up by its exact upstream name.
    public static func named(_ name: String) -> InstrumentGroup? {
        all.first { $0.name == name }
    }

    /// The group whose representative program this is, or nil.
    ///
    /// The model emits only representative programs, so this is how a decoded
    /// program token becomes a name. Anything else surfaces as `program_<n>` —
    /// including 96, which upstream's table accidentally aliases to drums via a
    /// singleton group that shares the drum class index. A rain-FX note is not
    /// a drum hit, so that alias is deliberately not reproduced.
    public static func forRepresentativeProgram(_ program: Int) -> InstrumentGroup? {
        if program == drumProgram { return named("drums") }
        return all.first { !$0.isDrums && $0.programs.first == program }
    }

    /// The label a decoded program gets: the group name, or `program_<n>`.
    public static func label(forProgram program: Int) -> String {
        forRepresentativeProgram(program)?.name ?? "program_\(program)"
    }

    // MARK: - Resolving loose input

    /// Resolve user-typed names to groups, the way upstream's CLI does.
    ///
    /// Case-insensitive; an exact name wins, otherwise a token that is a substring
    /// of exactly one name matches it (`"timp"` → timpani). Ambiguity and unknown
    /// names throw, the latter with close spellings suggested, because a silently
    /// dropped instrument would change what the model is allowed to decode.
    public static func resolve(_ tokens: [String]) throws -> [InstrumentGroup] {
        var resolved: [InstrumentGroup] = []
        for raw in tokens {
            let token = raw.trimmingCharacters(in: .whitespaces).lowercased()
            guard !token.isEmpty else { continue }
            if let exact = named(token) { resolved.append(exact); continue }
            let hits = all.filter { $0.name.contains(token) }
            switch hits.count {
            case 1: resolved.append(hits[0])
            case 0:
                let suggestions = all
                    .map { ($0.name, closeness(token, to: $0.name)) }
                    .filter { $0.1 >= 0.6 }
                    .sorted { $0.1 > $1.1 }
                    .prefix(3).map(\.0)
                throw MusicTranscriberError.unknownInstrument(raw, suggestions: Array(suggestions))
            default:
                throw MusicTranscriberError.ambiguousInstrument(raw, candidates: hits.map(\.name))
            }
        }
        return resolved
    }

    /// A similarity in 0…1 against the name and each of its underscore-separated
    /// words, so a typo like "pinao" still surfaces "acoustic_piano". Ratcliff/
    /// Obershelp, as in Python's difflib, so the suggestions match upstream's.
    static func closeness(_ token: String, to name: String) -> Double {
        let parts = [name] + name.split(separator: "_").map(String.init)
        return parts.map { ratio(Array(token), Array($0)) }.max() ?? 0
    }

    private static func ratio(_ a: [Character], _ b: [Character]) -> Double {
        guard !a.isEmpty, !b.isEmpty else { return 0 }
        return 2 * Double(matchingBlocks(a[...], b[...])) / Double(a.count + b.count)
    }

    private static func matchingBlocks(_ a: ArraySlice<Character>, _ b: ArraySlice<Character>) -> Int {
        guard !a.isEmpty, !b.isEmpty else { return 0 }
        var best = (i: a.startIndex, j: b.startIndex, length: 0)
        for i in a.indices {
            for j in b.indices {
                var k = 0
                while i + k < a.endIndex, j + k < b.endIndex, a[i + k] == b[j + k] { k += 1 }
                if k > best.length { best = (i, j, k) }
            }
        }
        guard best.length > 0 else { return 0 }
        return best.length
            + matchingBlocks(a[a.startIndex..<best.i], b[b.startIndex..<best.j])
            + matchingBlocks(a[(best.i + best.length)...], b[(best.j + best.length)...])
    }
}
