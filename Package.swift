// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MusicTranscriber",
    // Core AI (the model runtime) and MusicUnderstanding (the beat tracker, via
    // MusicAnalysis) both ship with the 27-era SDKs. There is no fallback path
    // for either, so the floor is honest rather than aspirational.
    platforms: [
        .macOS("27.0")
    ],
    products: [
        .library(name: "MusicTranscriber", targets: ["MusicTranscriber"]),
    ],
    dependencies: [
        .package(url: "https://github.com/arraypress/swift-midi-file.git", from: "0.4.0"),
        .package(url: "https://github.com/arraypress/swift-music-analysis.git", from: "0.3.0"),
    ],
    targets: [
        .target(
            name: "MusicTranscriber",
            dependencies: [
                .product(name: "MIDIFileKit", package: "swift-midi-file"),
                .product(name: "MusicAnalysis", package: "swift-music-analysis"),
            ]
        ),
        .testTarget(
            name: "MusicTranscriberTests",
            dependencies: ["MusicTranscriber"],
            resources: [
                .copy("Fixtures"),
            ]
        ),
    ]
)
