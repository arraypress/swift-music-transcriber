//
//  ModelVariant.swift
//  MusicTranscriber
//
//  Created by David Sherlock on 2026.
//
//  The three published MuScriptor checkpoints and what they cost.
//

import Foundation

/// A published MuScriptor checkpoint.
///
/// Every size shares one architecture, tokenizer and audio front end; only the
/// transformer's width and depth differ. The numbers below are the upstream
/// `config.json` values and the sizes of the converted `.aimodel` assets.
public enum ModelVariant: String, Codable, Sendable, CaseIterable {

    /// 14 layers, width 768, ~103M parameters. Fast, and measurably the weakest:
    /// on the dense demo clip it hit the 2,000-token budget on every chunk where
    /// `medium` finished in under 200.
    case small

    /// 24 layers, width 1024, ~307M parameters. The upstream default and this
    /// library's: 10 ms per token on an M3 Max, 1.2 GB in fp32.
    case medium

    /// 48 layers, width 1536, ~1.4B parameters. The most accurate; 5.5 GB in fp32.
    case large

    /// Transformer width.
    public var dimension: Int {
        switch self {
        case .small: return 768
        case .medium: return 1024
        case .large: return 1536
        }
    }

    /// Attention heads.
    public var heads: Int {
        switch self {
        case .small: return 12
        case .medium: return 16
        case .large: return 24
        }
    }

    /// Transformer layers.
    public var layers: Int {
        switch self {
        case .small: return 14
        case .medium: return 24
        case .large: return 48
        }
    }

    /// Output vocabulary width of the checkpoint.
    ///
    /// The tokenizer has 1,393 tokens; `medium` and `large` were trained with two
    /// spare rows, which the decoder masks out. See ``EventVocabulary/count``.
    public var card: Int {
        switch self {
        case .small: return 1393
        case .medium, .large: return 1395
        }
    }

    /// The Hugging Face repository the weights come from.
    public var repository: String { "MuScriptor/muscriptor-\(rawValue)" }

    /// The asset file name this library looks for, e.g. `scribe-medium-float32.aimodel`.
    public func assetName(precision: ModelPrecision) -> String {
        "scribe-\(rawValue)-\(precision.rawValue).aimodel"
    }
}

/// The precision a model asset was exported at.
///
/// Measured on the demo clip: fp32 reproduces the upstream CPU decode token for
/// token on every chunk; fp16 does too on `medium`, but on `small` — whose
/// output on that clip is already degenerate — it diverges after ~50 tokens,
/// exactly where upstream's own fp16-on-Metal run diverges. On this hardware
/// fp32 was no slower, so it is the default.
public enum ModelPrecision: String, Codable, Sendable, CaseIterable {
    case float32
    case float16
}
