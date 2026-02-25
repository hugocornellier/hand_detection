// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "hand_detection",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        .library(name: "hand-detection", targets: ["hand_detection"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "hand_detection",
            dependencies: [],
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
