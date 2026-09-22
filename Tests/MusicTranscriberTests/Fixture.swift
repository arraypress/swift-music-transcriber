//
//  Fixture.swift
//  MusicTranscriberTests
//
//  Created by David Sherlock on 2026.
//
//  Golden files dumped from the upstream Python code by Tools/dump_fixtures.py.
//

import Foundation
import XCTest

enum Fixture {
    static func url(_ name: String) throws -> URL {
        let parts = name.split(separator: ".")
        let url = Bundle.module.url(forResource: String(parts[0]), withExtension: String(parts[1]), subdirectory: "Fixtures")
        return try XCTUnwrap(url, "missing fixture \(name)")
    }

    static func floats(_ name: String) throws -> [Float] {
        try Data(contentsOf: url(name)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func json<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: url(name)))
    }

    /// Peak signal to noise, in dB, of `got` against `reference`.
    static func psnr(_ reference: [Float], _ got: [Float]) -> Double {
        var err = 0.0, peak = 0.0
        for i in 0..<reference.count {
            let d = Double(reference[i] - got[i]); err += d * d
            peak = max(peak, abs(Double(reference[i])))
        }
        let rms = (err / Double(reference.count)).squareRoot()
        return rms == 0 ? .infinity : 20 * log10(peak / rms)
    }
}
