//
//  PipeCollector.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  Reads a child process's pipe while it runs. A pipe holds about 64 KB:
//  MuseScore's stdout was read to its end before stderr, so a MuseScore
//  that wrote more than that to stderr stalled — and the timeout could
//  never fire while the read blocked. (CLIKit's ChildProcess does this for
//  CLIs; this library does not depend on CLIKit, so it keeps the few lines
//  it needs.)
//

import Foundation
import os

struct PipeCollector: Sendable {

    private let data = OSAllocatedUnfairLock(initialState: Data())
    private let finished = DispatchSemaphore(value: 0)

    /// Starts collecting `pipe` at once; call before running the child.
    init(_ pipe: Pipe) {
        let data = data, finished = finished
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                finished.signal()
            } else {
                data.withLock { $0.append(chunk) }
            }
        }
    }

    /// Everything the pipe carried, once it reaches its end — or what had
    /// arrived after `wait` seconds, if something still holds it open.
    func collected(wait: TimeInterval = 5) -> Data {
        _ = finished.wait(timeout: .now() + wait)
        return data.withLock { $0 }
    }
}
