// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "hand_detection_tflite",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        .library(name: "hand-detection-tflite", targets: ["hand_detection_tflite"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "hand_detection_tflite",
            dependencies: [],
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
