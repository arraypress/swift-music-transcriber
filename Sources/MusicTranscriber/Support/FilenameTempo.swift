//
//  FilenameTempo.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The tempo a sample-pack filename carries.
//
//  Loops are cut to a tempo and the packs say which: "VENDOR_PK2_123_bass_loop",
//  "Drums 128 Cmin", "bass_140bpm". On the loops measured for the README the
//  beat tracker locked onto a triplet subdivision and came back at 2/3 of the
//  true tempo on 8 of 23 misses; the name was right every time.
//

import Foundation

/// Reads a tempo out of a file name.
public enum FilenameTempo {

    /// Tempos a loop can plausibly be cut at. Outside this a number is a
    /// catalogue id, a bit depth or a year.
    public static let plausible = 60.0...220.0

    /// The BPM in `name`, or nil.
    ///
    /// A number followed by "bpm" wins outright. Otherwise the first standalone
    /// number — delimited by `_`, `-`, space, dot or the ends of the stem — that
    /// falls in ``plausible``. "PK2" is not standalone and "24" (a bit depth)
    /// is not plausible, so neither is mistaken for a tempo.
    public static func bpm(inFilename name: String) -> Double? {
        let stem = (name as NSString).deletingPathExtension
        if let match = stem.firstMatch(of: /(\d{2,3}(?:[.,]\d+)?)\s*[Bb][Pp][Mm]/),
           let value = Double(match.1.replacingOccurrences(of: ",", with: ".")), plausible.contains(value) {
            return value
        }
        for token in stem.split(whereSeparator: { "_- ." .contains($0) }) {
            guard let value = Double(token), plausible.contains(value) else { continue }
            return value
        }
        return nil
    }
}
