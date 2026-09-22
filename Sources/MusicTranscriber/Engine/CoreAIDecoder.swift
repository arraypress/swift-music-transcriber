//
//  CoreAIDecoder.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The model itself: one .aimodel, three functions, a KV cache that lives in
//  Core AI state. Everything above this file is tokens and floats.
//
//  The asset is produced by Tools/export.py. It carries `prefix` (log-mel and
//  instrument classes to the conditioning prefix), `embed` (token ids to
//  embeddings) and `main` (embeddings in, next-token logits out, with the
//  attention cache as mutable state). The engine is loaded GPU-first with
//  `expectFrequentReshapes` on: without that flag the runtime re-specialises
//  the graph for every new sequence length, ~180 ms a step, and — measured —
//  its specialised decode path drifted to 26 dB against PyTorch where the
//  reshape-tolerant path holds 68 dB.
//

import CoreAI
import Foundation

/// A loaded MuScriptor model on Core AI.
public final class CoreAIDecoder: @unchecked Sendable {

    /// The asset this was loaded from.
    public let url: URL

    /// Transformer width, read off the asset.
    public let dimension: Int

    /// Layers, heads and the cache's context length, read off the asset.
    public let layers: Int
    public let heads: Int
    public let maxContext: Int

    /// Whether the transformer runs in half precision.
    public let isHalf: Bool

    /// Width of the logits, and therefore the id of the "start of chunk" token,
    /// which sits one past the last vocabulary row. 1,393 for `small`, 1,395 for
    /// the others; read off the asset so a future checkpoint cannot lie.
    public let card: Int

    private let main: InferenceFunction
    private let prefix: InferenceFunction
    private let embed: InferenceFunction
    private let keyDescriptor: NDArrayDescriptor
    private let valueDescriptor: NDArrayDescriptor

    /// Load an asset. The first load of a given asset compiles for this GPU and
    /// takes a few seconds; the runtime caches that, so later loads take ~0.3 s.
    public init(contentsOf url: URL) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MusicTranscriberError.modelNotFound(url.path)
        }
        self.url = url
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model: AIModel
        do {
            model = try await AIModel(contentsOf: url, options: options)
        } catch {
            throw MusicTranscriberError.inferenceFailed("could not load \(url.lastPathComponent): \(error)")
        }
        guard let main = try model.loadFunction(named: "main"),
              let prefix = try model.loadFunction(named: "prefix"),
              let embed = try model.loadFunction(named: "embed") else {
            throw MusicTranscriberError.modelIncompatible(
                "expected functions main, prefix and embed; found \(model.functionNames)")
        }
        guard case .ndArray(let k) = main.descriptor.stateDescriptor(of: "k_cache"),
              case .ndArray(let v) = main.descriptor.stateDescriptor(of: "v_cache"),
              k.shape.count == 5 else {
            throw MusicTranscriberError.modelIncompatible("main has no k_cache / v_cache state")
        }
        self.main = main
        self.prefix = prefix
        self.embed = embed
        keyDescriptor = k
        valueDescriptor = v
        layers = k.shape[0]
        heads = k.shape[2]
        maxContext = k.shape[3]
        dimension = k.shape[2] * k.shape[4]
        isHalf = k.scalarType == .float16
        guard case .ndArray(let logits) = main.descriptor.outputDescriptor(of: "logits"),
              let width = logits.shape.last, width > 0 else {
            throw MusicTranscriberError.modelIncompatible("main has no logits output with a fixed width")
        }
        card = width
    }

    /// The published variant with these dimensions, if it is one.
    public var variant: ModelVariant? {
        ModelVariant.allCases.first { $0.layers == layers && $0.dimension == dimension }
    }

    // MARK: - State

    /// One attention cache: the model's memory of the current chunk.
    public final class State {
        var key: NDArray
        var value: NDArray
        init(key: NDArray, value: NDArray) { self.key = key; self.value = value }
    }

    /// A fresh, zeroed cache.
    public func makeState() -> State {
        var k = NDArray(descriptor: keyDescriptor)
        var v = NDArray(descriptor: valueDescriptor)
        Self.zero(&k); Self.zero(&v)
        return State(key: k, value: v)
    }

    /// Zero a cache in place, for the next chunk.
    public func reset(_ state: State) {
        Self.zero(&state.key); Self.zero(&state.value)
    }

    /// Copy `source`'s cache into `destination` — beam search reordering.
    public func copy(_ source: State, into destination: State) {
        Self.copyBytes(from: source.key, to: &destination.key)
        Self.copyBytes(from: source.value, to: &destination.value)
    }

    // MARK: - Functions

    /// Conditioning prefix for one chunk: `[mel(501), dataset(1), instruments(n)]`
    /// as `count × dimension` floats.
    ///
    /// `instrumentTokens` are class ids plus one, or `[0]` for "unconditional".
    /// A null mel condition — `mel` of one all-zero frame — produces the zero
    /// prefix classifier-free guidance needs.
    public func prefixEmbeddings(mel: [Float], frames: Int, instrumentTokens: [Int32]) async throws -> (values: [Float], count: Int) {
        var outputs = try await prefix.run(inputs: [
            "mel": Self.array(mel, shape: [1, frames, MelSpectrogram.mels], half: false),
            "inst_tokens": Self.array(instrumentTokens, shape: [1, instrumentTokens.count]),
        ])
        guard let out = outputs.remove("embeds")?.ndArray else {
            throw MusicTranscriberError.inferenceFailed("prefix produced no embeds")
        }
        return (Self.floats(out), out.shape[1])
    }

    /// Token embeddings, `tokens.count × dimension` floats.
    public func embeddings(of tokens: [Int32]) async throws -> [Float] {
        var outputs = try await embed.run(inputs: ["tokens": Self.array(tokens, shape: [1, tokens.count])])
        guard let out = outputs.remove("embeds")?.ndArray else {
            throw MusicTranscriberError.inferenceFailed("embed produced no embeds")
        }
        return Self.floats(out)
    }

    /// Run the transformer over `count` new positions whose embeddings are
    /// `values`, given that `sequenceLength - count` positions are already in the
    /// cache. Returns the logits of the last position.
    public func logits(embeddings values: [Float], count: Int, sequenceLength: Int, state: State) async throws -> [Float] {
        precondition(values.count == count * dimension)
        precondition(sequenceLength <= maxContext, "sequence exceeds the cache")
        let inputs: [String: NDArray] = [
            "inputs_embeds": Self.array(values, shape: [1, count, dimension], half: isHalf),
            "position_ids": Self.array((0..<Int32(sequenceLength)).map { $0 }, shape: [1, sequenceLength]),
        ]
        // The cache views borrow the arrays for the duration of the call; going
        // through inout parameters is what lets the lifetime checker see that.
        return try await run(inputs: inputs, key: &state.key, value: &state.value)
    }

    private func run(inputs: [String: NDArray], key: inout NDArray, value: inout NDArray) async throws -> [Float] {
        var states = InferenceFunction.MutableViews()
        states.insert(key.mutableRawView(), for: "k_cache")
        states.insert(value.mutableRawView(), for: "v_cache")
        var outputs: InferenceFunction.Outputs
        do {
            outputs = try await main.run(inputs: inputs, states: consume states, outputViews: InferenceFunction.MutableViews())
        } catch {
            throw MusicTranscriberError.inferenceFailed("\(error)")
        }
        guard let out = outputs.remove("logits")?.ndArray else {
            throw MusicTranscriberError.inferenceFailed("main produced no logits")
        }
        return Self.floats(out)
    }

    // MARK: - NDArray plumbing

    static func array(_ values: [Float], shape: [Int], half: Bool) -> NDArray {
        if half {
            var a = NDArray(shape: shape, scalarType: .float16)
            var view = a.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { p, _, _ in
                for i in 0..<values.count { p[i] = Float16(values[i]) }
            }
            return a
        }
        var a = NDArray(shape: shape, scalarType: .float32)
        var view = a.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { p, _, _ in
            values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) }
        }
        return a
    }

    static func array(_ values: [Int32], shape: [Int]) -> NDArray {
        var a = NDArray(shape: shape, scalarType: .int32)
        var view = a.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { p, _, _ in
            values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) }
        }
        return a
    }

    static func floats(_ a: NDArray) -> [Float] {
        let count = a.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        switch a.scalarType {
        case .float16:
            a.view(as: Float16.self).withUnsafePointer { p, _, _ in for i in 0..<count { out[i] = Float(p[i]) } }
        case .float32:
            a.view(as: Float.self).withUnsafePointer { p, _, _ in for i in 0..<count { out[i] = p[i] } }
        default:
            preconditionFailure("unexpected scalar type \(a.scalarType)")
        }
        return out
    }

    static func zero(_ a: inout NDArray) {
        let bytes = a.shape.reduce(1, *) * (a.scalarType == .float32 ? 4 : 2)
        a.mutableRawView().withUnsafeMutableBytes { p, _, _ in memset(p, 0, bytes) }
    }

    static func copyBytes(from source: NDArray, to destination: inout NDArray) {
        let bytes = source.shape.reduce(1, *) * (source.scalarType == .float32 ? 4 : 2)
        source.rawView().withUnsafeBytes { src, _, _ in
            destination.mutableRawView().withUnsafeMutableBytes { dst, _, _ in memcpy(dst, src, bytes) }
        }
    }
}
