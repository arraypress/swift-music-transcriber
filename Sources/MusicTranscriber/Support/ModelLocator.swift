//
//  ModelLocator.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Where a model asset is looked for, in the order a per-run choice should
//  beat a per-shell one should beat the installed copy.
//

import Foundation

/// Finds model assets.
///
/// The assets are not bundled: the weights are CC BY-NC 4.0 and up to 5.5 GB,
/// so they are converted once with `Tools/export.py` and installed into
/// Application Support. `--model` and `$SCRIBE_MODEL` override per run and
/// per shell.
public enum ModelLocator {

    /// The environment variable that names a model directory or asset.
    public static let environmentVariable = "SCRIBE_MODEL"

    /// `~/Library/Application Support/scribe/models`.
    public static var installDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("scribe/models", isDirectory: true)
    }

    /// Resolve an asset.
    ///
    /// - Parameters:
    ///   - explicit: a path to an `.aimodel`, or to a directory of them, if the caller gave one.
    ///   - variant: which size to pick from a directory.
    ///   - precision: which precision to pick from a directory; the other is
    ///     accepted when only it is installed.
    ///   - environment: the process environment, injectable for tests.
    /// - Throws: ``MusicTranscriberError/modelNotFound(_:)`` naming every place looked.
    ///   An explicit path that has nothing at it is an error on its own: a per-run
    ///   choice that silently fell back to the installed copy would be the wrong
    ///   model with no warning.
    public static func resolve(explicit: String? = nil, variant: ModelVariant = .medium,
                               precision: ModelPrecision = .float32,
                               environment: [String: String] = ProcessInfo.processInfo.environment) throws -> URL {
        var looked: [String] = []
        var roots: [URL] = []
        if let explicit, !explicit.isEmpty {
            roots.append(URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath))
        }
        if let env = environment[environmentVariable], !env.isEmpty {
            roots.append(URL(fileURLWithPath: (env as NSString).expandingTildeInPath))
        }
        roots.append(installDirectory)

        for (i, root) in roots.enumerated() {
            let isExplicit = explicit != nil && i == 0
            if root.pathExtension == "aimodel" {
                looked.append(root.path)
                if FileManager.default.fileExists(atPath: root.path) { return root }
                if isExplicit { throw MusicTranscriberError.modelNotFound(root.path) }
                continue
            }
            if isExplicit, !FileManager.default.fileExists(atPath: root.path) {
                throw MusicTranscriberError.modelNotFound(root.path)
            }
            for p in [precision] + ModelPrecision.allCases.filter({ $0 != precision }) {
                let candidate = root.appendingPathComponent(variant.assetName(precision: p))
                looked.append(candidate.path)
                if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            }
        }
        throw MusicTranscriberError.modelNotFound("no \(variant.rawValue) model (looked at \(looked.joined(separator: ", ")))")
    }

    /// Resolve the Beat This! tracker asset: an explicit path, `$SCRIBE_BEAT_THIS`,
    /// then ``installDirectory``. Throws when none is installed; callers fall
    /// back to Apple's tracker and say so.
    public static func resolveBeatTracker(explicit: String? = nil,
                                          environment: [String: String] = ProcessInfo.processInfo.environment) throws -> URL {
        var looked: [String] = []
        var candidates: [URL] = []
        if let explicit, !explicit.isEmpty { candidates.append(URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)) }
        if let env = environment["SCRIBE_BEAT_THIS"], !env.isEmpty { candidates.append(URL(fileURLWithPath: (env as NSString).expandingTildeInPath)) }
        candidates.append(installDirectory.appendingPathComponent(BeatThisTracker.assetName))
        for candidate in candidates {
            let url = candidate.pathExtension == "aimodel" ? candidate : candidate.appendingPathComponent(BeatThisTracker.assetName)
            looked.append(url.path)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        throw MusicTranscriberError.modelNotFound("no beat tracker (looked at \(looked.joined(separator: ", ")))")
    }

    /// The assets installed in ``installDirectory``, with their sizes in bytes.
    public static func installed(in directory: URL = installDirectory) -> [(url: URL, bytes: Int64)] {
        guard let items = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            return []
        }
        return items.filter { $0.pathExtension == "aimodel" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { ($0, size(of: $0)) }
    }

    /// Bytes under an asset (assets are directories).
    public static func size(of url: URL) -> Int64 {
        guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in walker {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }
}
