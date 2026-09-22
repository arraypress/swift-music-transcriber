//
//  SheetMusic.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Engraved sheet music: MIDI → MusicXML and PDFs, through the MuseScore CLI.
//
//  MuseScore is an external program, not a dependency: this only works on a
//  machine that has MuseScore 4 or newer installed. Everything here shells out
//  to it and then checks that the files it was asked for actually appeared —
//  `mscore` exits 0 when it writes nothing at all. MuseScore 3 is rejected: its
//  project format carries no string data for guitars, so the tab conversion
//  would quietly produce a score with no tablature in it.
//

import Foundation

/// Sheet-music export via MuseScore, a port of upstream's `utils/sheets.py`.
public enum SheetMusic {

    /// Checked before anything else, for a MuseScore that isn't on PATH.
    public static let environmentVariable = "SCRIBE_MUSESCORE"

    /// MuseScore 3 and older are rejected; see the file comment.
    public static let minimumMajorVersion = 4

    static let binaryNames = ["mscore", "musescore", "mscore4portable", "MuseScore4", "musescore4", "mscore3", "musescore3"]
    static let appLocations = [
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
        "~/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    ]
    static let installHint = "Downloads for every platform: https://musescore.org/en/download\nIf it is installed somewhere unusual, set $\(environmentVariable) to it."

    /// Per invocation. Generous next to the ~1.5 s a song takes; a wedged
    /// MuseScore fails the run instead of hanging it.
    static let timeout: TimeInterval = 120

    /// Tablature presets by string count; MuseScore ships tab4Str…tab9Str.
    static func tabPreset(strings: Int) -> String? { (4...9).contains(strings) ? "tab\(strings)StrCommon" : nil }

    // MARK: - Finding it

    /// Path to a MuseScore 4+ executable.
    ///
    /// `$SCRIBE_MUSESCORE` first, then PATH, then the places the download puts
    /// it. Names any too-old MuseScore it did find, since "not found" is a
    /// confusing thing to read with `mscore` on your PATH.
    public static func findMuseScore(environment: [String: String] = ProcessInfo.processInfo.environment) throws -> URL {
        if let override = environment[environmentVariable], !override.isEmpty {
            let url = URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
            guard FileManager.default.isExecutableFile(atPath: url.path) else {
                throw MusicTranscriberError.externalToolMissing("MuseScore", hint: "\(environmentVariable) is set to \(override), which is not an executable file.")
            }
            guard let version = version(of: url), version[0] >= minimumMajorVersion else {
                throw MusicTranscriberError.externalToolMissing("MuseScore", hint: "\(environmentVariable) points at an old or unrecognised MuseScore; sheets need MuseScore \(minimumMajorVersion) or newer.")
            }
            return url
        }
        var tooOld: [String] = []
        for candidate in candidates(environment: environment) {
            guard let version = version(of: candidate) else { continue }
            if version[0] >= minimumMajorVersion { return candidate }
            tooOld.append("\(candidate.path) (MuseScore \(version.map(String.init).joined(separator: ".")))")
        }
        if !tooOld.isEmpty {
            throw MusicTranscriberError.externalToolMissing(
                "MuseScore \(minimumMajorVersion)+",
                hint: "the only MuseScore installed is:\n  " + tooOld.joined(separator: "\n  ")
                    + "\nMuseScore 3 cannot produce the guitar and bass tablature.\n" + installHint)
        }
        throw MusicTranscriberError.externalToolMissing(
            "MuseScore", hint: "sheets are engraved with MuseScore \(minimumMajorVersion)+, which has to be installed separately.\n" + installHint)
    }

    static func candidates(environment: [String: String]) -> [URL] {
        var found: [URL] = []
        let path = (environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for name in binaryNames {
            for dir in path {
                let url = URL(fileURLWithPath: dir).appendingPathComponent(name)
                if FileManager.default.isExecutableFile(atPath: url.path) { found.append(url) }
            }
        }
        for location in appLocations {
            let url = URL(fileURLWithPath: (location as NSString).expandingTildeInPath)
            if FileManager.default.isExecutableFile(atPath: url.path) { found.append(url) }
        }
        return found
    }

    /// `mscore --version` prints e.g. "MuseScore4 4.7.4"; the version as integers, or nil.
    static func version(of binary: URL) -> [Int]? {
        guard let result = try? run(binary, ["--version"]) else { return nil }
        let text = result.stdout + "\n" + result.stderr
        guard let match = text.firstMatch(of: /(\d+)\.(\d+)(?:\.(\d+))?/) else { return nil }
        return [Int(match.1)!, Int(match.2)!] + (match.3.flatMap { Int($0) }.map { [$0] } ?? [])
    }

    // MARK: - Writing

    /// Check `directory` can take a score: absent, or an existing empty directory.
    ///
    /// A run must not scatter PDFs among unrelated files or quietly overwrite a
    /// previous score. The directory itself is created by ``write(midi:to:musescore:quantized:)``.
    public static func prepareOutputDirectory(_ directory: URL) throws {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDir) else { return }
        guard isDir.boolValue else { throw MusicTranscriberError.invalidOptions("\(directory.path) exists and is not a directory") }
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: directory.path), !contents.isEmpty {
            throw MusicTranscriberError.invalidOptions("\(directory.path) is not empty")
        }
    }

    /// Engrave `midi` into `directory`, returning the files written.
    ///
    /// The MIDI, a MusicXML score, one PDF of the full score, one PDF of
    /// notation per instrument, and for guitar and bass parts a tablature PDF
    /// rendered from a second copy of the score whose staves are retyped as tab.
    ///
    /// `quantized` says whether the notes are already on a beat grid, which
    /// decides the triplet search: unquantized input engraves timing jitter as
    /// tied 128th notes.
    public static func write(midi: Data, to directory: URL, musescore: URL? = nil, quantized: Bool = false) throws -> [URL] {
        let binary = try musescore ?? findMuseScore()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let midiURL = directory.appendingPathComponent("score.mid")
        try midi.write(to: midiURL)
        var written = [midiURL]

        let temp = FileManager.default.temporaryDirectory.appendingPathComponent("scribe-sheets-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }

        let options = temp.appendingPathComponent("import.xml")
        try importOptions(quantized: quantized).write(to: options, atomically: true, encoding: .utf8)

        // MuseScore's own project format, so the score can be copied and edited
        // (staves retyped as tab) before anything is rendered from it.
        let mscx = temp.appendingPathComponent("score.mscx")
        var result = try run(binary, ["-M", options.path, "-o", mscx.path, midiURL.path])
        try ensure(mscx, "import the MIDI file", result)

        let musicxml = directory.appendingPathComponent("score.musicxml")
        result = try run(binary, ["-o", musicxml.path, mscx.path])
        try ensure(musicxml, "write MusicXML", result)
        written.append(musicxml)

        let full = directory.appendingPathComponent("full_score.pdf")
        result = try run(binary, ["-o", full.path, mscx.path])
        try ensure(full, "render the full score", result)
        written.append(full)

        written += try partPDFs(binary, mscx: mscx, into: directory)

        let fretted = try frettedParts(mscx)
        if !fretted.isEmpty {
            let tab = temp.appendingPathComponent("tab.mscx")
            try FileManager.default.copyItem(at: mscx, to: tab)
            try convertToTabStaves(tab)
            written += try partPDFs(binary, mscx: tab, into: directory, suffix: "_tab", only: Set(fretted))
        }
        return written
    }

    /// MuseScore's MIDI import settings. QuantValue 2 is 1/16; triplets only when
    /// the grid is known to be accurate.
    static func importOptions(quantized: Bool) -> String {
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <MidiOptions>
          <QuantValue>2</QuantValue>
          <HumanPerformance>true</HumanPerformance>
          <Duplets>false</Duplets>
          <Triplets>\(quantized ? "true" : "false")</Triplets>
          <Quadruplets>false</Quadruplets>
          <Quintuplets>false</Quintuplets>
          <Septuplets>false</Septuplets>
          <Nonuplets>false</Nonuplets>
          <SimplifyDurations>true</SimplifyDurations>
          <DottedNotes>true</DottedNotes>
        </MidiOptions>

        """
    }

    /// One PDF per instrument, from `--score-parts-pdf`, which hands the parts
    /// back as base64 in a JSON blob on stdout rather than writing files.
    static func partPDFs(_ binary: URL, mscx: URL, into directory: URL, suffix: String = "", only: Set<Int>? = nil) throws -> [URL] {
        let result = try run(binary, ["--score-parts-pdf", mscx.path])
        guard let json = try? JSONSerialization.jsonObject(with: Data(result.stdout.utf8)) as? [String: Any],
              let names = json["parts"] as? [String], let blobs = json["partsBin"] as? [String] else {
            throw failure("generate the per-instrument PDFs", result)
        }
        var written: [URL] = []
        for (index, (name, blob)) in zip(names, blobs).enumerated() {
            if let only, !only.contains(index) { continue }
            guard let data = Data(base64Encoded: blob) else { continue }
            let url = directory.appendingPathComponent(String(format: "%02d_%@%@.pdf", index + 1, slug(name), suffix))
            try data.write(to: url)
            written.append(url)
        }
        guard !written.isEmpty else { throw failure("generate the per-instrument PDFs", result) }
        return written
    }

    /// Filename stem for a part, with the repetition in its name dropped.
    ///
    /// MuseScore names a part after the instrument it matched, then appends the
    /// track name the MIDI carried, which usually says the same thing twice:
    /// "Electric Guitar, clean electric guitar". A segment whose words all
    /// appear in another adds nothing and goes.
    static func slug(_ name: String) -> String {
        let segments = name.split(separator: ",").map(normalize).filter { !$0.isEmpty }
        let words = segments.map { Set($0.split(separator: "_").map(String.init)) }
        let kept = segments.indices.filter { i in
            !words.indices.contains { j in
                j != i && (words[i].isStrictSubset(of: words[j]) || (words[i] == words[j] && j < i))
            }
        }.map { segments[$0] }
        return kept.isEmpty ? "part" : kept.joined(separator: "_")
    }

    static func normalize(_ text: Substring) -> String {
        text.replacing(/[^A-Za-z0-9]+/, with: "_").trimmingCharacters(in: CharacterSet(charactersIn: "_")).lowercased()
    }

    /// Indices of the parts whose instrument has a string count with a tab preset.
    static func frettedParts(_ mscx: URL) throws -> [Int] {
        try stringCounts(mscx).enumerated().compactMap { tabPreset(strings: $0.element) != nil ? $0.offset : nil }
    }

    static func stringCounts(_ mscx: URL) throws -> [Int] {
        let document = try XMLDocument(contentsOf: mscx, options: [])
        guard let score = try document.nodes(forXPath: "/museScore/Score").first as? XMLElement else {
            throw MusicTranscriberError.inferenceFailed("\(mscx.lastPathComponent) has no <Score> element")
        }
        return score.elements(forName: "Part").map { part in
            part.elements(forName: "Instrument").first?.elements(forName: "StringData").first?.elements(forName: "string").count ?? 0
        }
    }

    /// Retype every fretted part's staff as tablature, in place.
    static func convertToTabStaves(_ mscx: URL) throws {
        let document = try XMLDocument(contentsOf: mscx, options: [])
        guard let score = try document.nodes(forXPath: "/museScore/Score").first as? XMLElement else {
            throw MusicTranscriberError.inferenceFailed("\(mscx.lastPathComponent) has no <Score> element")
        }
        for part in score.elements(forName: "Part") {
            let strings = part.elements(forName: "Instrument").first?.elements(forName: "StringData").first?.elements(forName: "string").count ?? 0
            guard let preset = tabPreset(strings: strings), let staff = part.elements(forName: "Staff").first else { continue }
            let type = staff.elements(forName: "StaffType").first ?? {
                let e = XMLElement(name: "StaffType"); staff.addChild(e); return e
            }()
            type.removeAttribute(forName: "group")
            type.addAttribute(XMLNode.attribute(withName: "group", stringValue: "tablature") as! XMLNode)
            let name = type.elements(forName: "name").first ?? { let e = XMLElement(name: "name"); type.addChild(e); return e }()
            name.stringValue = preset
            // <name> is only a label to MuseScore's reader; the geometry must be
            // spelled out or a 6-string guitar lands on the default 5 lines.
            for tag in ["lines", "lineDistance"] {
                for existing in type.elements(forName: tag) { existing.detach() }
            }
            type.addChild(XMLElement(name: "lines", stringValue: String(strings)))
            type.addChild(XMLElement(name: "lineDistance", stringValue: "1.5"))
        }
        try document.xmlData(options: [.nodePrettyPrint]).write(to: mscx)
    }

    // MARK: - Running it

    struct Completed { let status: Int32; let stdout: String; let stderr: String }

    /// MuseScore with `arguments`, headless. The status is deliberately not
    /// checked here: it exits 0 for several failures that write no file.
    static func run(_ binary: URL, _ arguments: [String]) throws -> Completed {
        let process = Process()
        process.executableURL = binary
        process.arguments = arguments
        let out = Pipe(), err = Pipe()
        process.standardOutput = out
        process.standardError = err
        try process.run()
        let deadline = DispatchTime.now() + timeout
        let group = DispatchGroup()
        group.enter()
        process.terminationHandler = { _ in group.leave() }
        let outData = out.fileHandleForReading.readDataToEndOfFile()
        let errData = err.fileHandleForReading.readDataToEndOfFile()
        if group.wait(timeout: deadline) == .timedOut {
            process.terminate()
            return Completed(status: 1, stdout: "", stderr: "timed out after \(Int(timeout))s")
        }
        return Completed(status: process.terminationStatus,
                         stdout: String(decoding: outData, as: UTF8.self),
                         stderr: String(decoding: errData, as: UTF8.self))
    }

    static func ensure(_ file: URL, _ what: String, _ result: Completed) throws {
        guard FileManager.default.fileExists(atPath: file.path) else { throw failure(what, result) }
    }

    static func failure(_ what: String, _ result: Completed) -> MusicTranscriberError {
        let tail = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .split(separator: "\n").suffix(5).joined(separator: "\n  ")
        return .inferenceFailed("MuseScore failed to \(what)." + (tail.isEmpty ? "" : "\n  " + tail))
    }
}
