<h1 align="center">hand_detection</h1>

<p align="center">
<a href="https://flutter.dev"><img src="https://img.shields.io/badge/Platform-Flutter-02569B?logo=flutter" alt="Platform"></a>
<a href="https://dart.dev"><img src="https://img.shields.io/badge/language-Dart-blue" alt="Language: Dart"></a>
<br>
<a href="https://pub.dev/packages/hand_detection"><img src="https://img.shields.io/pub/v/hand_detection?label=pub.dev&labelColor=333940&logo=dart" alt="Pub Version"></a>
<a href="https://pub.dev/packages/hand_detection/score"><img src="https://img.shields.io/pub/points/hand_detection?color=2E8B57&label=pub%20points" alt="pub points"></a>
<a href="https://github.com/hugocornellier/hand_detection/actions/workflows/build.yml"><img src="https://github.com/hugocornellier/hand_detection/actions/workflows/build.yml/badge.svg" alt="CI"></a>
<a href="https://github.com/hugocornellier/hand_detection/actions/workflows/integration.yml"><img src="https://github.com/hugocornellier/hand_detection/actions/workflows/integration.yml/badge.svg" alt="Tests"></a>
<a href="https://github.com/hugocornellier/hand_detection/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-007A88.svg?logo=apache" alt="License"></a>
</p>

Flutter implementation of Google's MediaPipe hand detection and landmark models using TensorFlow Lite.
Completely local: no remote API, just pure on-device, offline detection.
 
<p align="center">
  <img src="assets/screenshots/hand-detection-demo.webp" alt="Hand detection example" width="640">
</p>

## Features

- On-device hand detection, runs fully offline
- 21-point hand landmarks with **3D depth information** (x, y, z coordinates)
- Handedness detection (left/right hand)
- **Gesture recognition**: closed fist, open palm, pointing up, thumbs down, thumbs up, victory, I love you
- Truly cross-platform: compatible with Android, iOS, macOS, Windows, and Linux
- The [example](https://pub.dev/packages/hand_detection/example) app illustrates how to detect and render results on images

## Quick Start

```dart
import 'dart:io';
import 'package:hand_detection/hand_detection.dart';

Future main() async {
  final detector = await HandDetector.create();

  final imageBytes = await File('path/to/image.jpg').readAsBytes();
  List<Hand> hands = await detector.detect(imageBytes);

  for (final hand in hands) {
    final boundingBox = hand.boundingBox;
    final handedness = hand.handedness;

    if (hand.hasLandmarks) {
      final wrist = hand.getLandmark(HandLandmarkType.wrist);
      final indexTip = hand.getLandmark(HandLandmarkType.indexFingerTip);
      print('Wrist: (${wrist?.x}, ${wrist?.y})');
    }
  }

  await detector.dispose();
}
```

## Performance

### Hardware Acceleration

`HandDetector` runs on one of two inference engines, selected at init:

- **Interpreter** (default). Classic TFLite. CPU via XNNPACK on every platform. GPU only via the platform delegates below, which are deprecated and platform-limited.
- **CompiledModel** (opt-in: `useCompiledModel: true`). LiteRT Next. Auto-selects GPU/NPU with automatic CPU fallback on every platform, and it is faster on CPU too (parity-checked: roughly 1.4x to 3.5x vs the plain Interpreter, at or above XNNPACK on most models).

| Platform | Interpreter GPU (default engine) | CompiledModel GPU (`useCompiledModel: true`) |
|----------|:---:|:---:|
| Android | ✅ `GpuDelegateV2`* | ✅ |
| iOS / macOS | ✅ Metal* | ✅ |
| **Windows / Linux** | ❌ CPU only (XNNPACK) | ✅ |
| Web | WebGPU via `liteRtAccelerator` | (n/a) |

> \*Interpreter GPU/Metal delegates are deprecated (removed in flutter_litert 4.0.0). **On Windows and Linux, GPU is available only through CompiledModel**, because the Interpreter has no desktop GPU delegate.

```dart
// Default (Interpreter): CPU everywhere; GPU on Android and Apple only.
final detector = await HandDetector.create();

// CompiledModel: GPU/NPU where available, automatic CPU fallback.
// This is the only GPU path on Windows and Linux.
final detector = await HandDetector.create(useCompiledModel: true);
```

### Accelerator selection (CompiledModel)

When `useCompiledModel: true`, two optional parameters control the LiteRT Next backend. They have no effect on the default Interpreter engine.

- `accelerators` (`Set<Accelerator>`, default `{Accelerator.gpu, Accelerator.cpu}`). The accelerators the backend may use. The runtime picks the fastest available and falls back through the set. If none initialize it throws, so include `Accelerator.cpu` to guarantee a fallback. The default requests GPU with CPU fallback.
- `precision` (`Precision`, default `Precision.fp16`). Numeric precision for the compiled graph. `Precision.fp32` trades speed for accuracy.

```dart
// GPU with automatic CPU fallback (the default).
await HandDetector.create(useCompiledModel: true);

// CPU only, using CompiledModel's fast CPU runtime.
await HandDetector.create(
  useCompiledModel: true,
  accelerators: {Accelerator.cpu},
);

// GPU only. Throws if the GPU backend cannot initialize.
await HandDetector.create(
  useCompiledModel: true,
  accelerators: {Accelerator.gpu},
);

// NPU first, CPU fallback.
await HandDetector.create(
  useCompiledModel: true,
  accelerators: {Accelerator.npu, Accelerator.cpu},
);

// Full fp32 precision.
await HandDetector.create(
  useCompiledModel: true,
  precision: Precision.fp32,
);
```

`Accelerator` and `Precision` are exported from the package.

### Advanced Performance Configuration

`performanceConfig` tunes the **Interpreter** engine only. It has no effect when `useCompiledModel: true`.

```dart
// Auto mode (default), optimal for each platform
final detector = await HandDetector.create();

// Force XNNPACK (all native platforms)
final detector = await HandDetector.create(
  performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
);

// Force the Interpreter GPU delegate (Android and Apple only; deprecated, prefer CompiledModel)
final detector = await HandDetector.create(
  performanceConfig: PerformanceConfig.gpu(),
);

// CPU-only (maximum compatibility)
final detector = await HandDetector.create(
  performanceConfig: PerformanceConfig.disabled,
);
```

### Advanced: Direct Mat Input

If you already have a decoded `cv.Mat` from another OpenCV pipeline, pass it directly:

```dart
import 'package:hand_detection/hand_detection.dart';

Future<void> processFrame(Mat frame) async {
  final detector = await HandDetector.create();

  final hands = await detector.detectFromMat(frame);

  frame.dispose(); // always dispose Mats after use
  await detector.dispose();
}
```

For live camera streams, prefer `prepareCameraFrame` + `detectFromCameraFrame` (see below): it keeps `cvtColor` / `rotate` / downscale off the UI thread.

## Bounding Boxes

The boundingBox property returns a BoundingBox object representing the hand bounding box in
absolute pixel coordinates. The BoundingBox provides convenient access to corner points,
dimensions (width and height), and the center point.

### Accessing Corners

```dart
final BoundingBox boundingBox = hand.boundingBox;

// Access individual corners by name (each is a Point with x and y)
final Point topLeft     = boundingBox.topLeft;       // Top-left corner
final Point topRight    = boundingBox.topRight;      // Top-right corner
final Point bottomRight = boundingBox.bottomRight;   // Bottom-right corner
final Point bottomLeft  = boundingBox.bottomLeft;    // Bottom-left corner

// Access coordinates
print('Top-left: (${topLeft.x}, ${topLeft.y})');
```

### Additional Bounding Box Parameters

```dart
final BoundingBox boundingBox = hand.boundingBox;

// Access dimensions and center
final double width  = boundingBox.width;     // Width in pixels
final double height = boundingBox.height;    // Height in pixels
final Point center = boundingBox.center;  // Center point

// Access coordinates
print('Size: ${width} x ${height}');
print('Center: (${center.x}, ${center.y})');

// Access all corners as a list (order: top-left, top-right, bottom-right, bottom-left)
final List<Point> allCorners = boundingBox.corners;
```

## Hand Landmarks (21-Point)

The `landmarks` property returns a list of 21 `HandLandmark` objects representing key points
on the detected hand. Each landmark has 3D coordinates (x, y, z) and a visibility score.

### 21 Hand Landmarks

| Index | Landmark | Description |
|-------|----------|-------------|
| 0 | wrist | Wrist |
| 1-4 | thumbCMC, thumbMCP, thumbIP, thumbTip | Thumb joints and tip |
| 5-8 | indexFingerMCP, indexFingerPIP, indexFingerDIP, indexFingerTip | Index finger |
| 9-12 | middleFingerMCP, middleFingerPIP, middleFingerDIP, middleFingerTip | Middle finger |
| 13-16 | ringFingerMCP, ringFingerPIP, ringFingerDIP, ringFingerTip | Ring finger |
| 17-20 | pinkyMCP, pinkyPIP, pinkyDIP, pinkyTip | Pinky finger |

### Accessing Landmarks

```dart
final Hand hand = hands.first;

// Access specific landmarks by type
final wrist = hand.getLandmark(HandLandmarkType.wrist);
final indexTip = hand.getLandmark(HandLandmarkType.indexFingerTip);
final thumbTip = hand.getLandmark(HandLandmarkType.thumbTip);

if (wrist != null) {
  print('Wrist: (${wrist.x}, ${wrist.y}, ${wrist.z})');
  print('Visibility: ${wrist.visibility}');
}

// Iterate through all landmarks
for (final landmark in hand.landmarks) {
  print('${landmark.type.name}: (${landmark.x}, ${landmark.y})');
}
```

### Drawing Hand Skeleton

Use the `handLandmarkConnections` constant to draw the hand skeleton:

```dart
import 'package:hand_detection/hand_detection.dart';

// Draw skeleton connections
for (final connection in handLandmarkConnections) {
  final start = hand.getLandmark(connection[0]);
  final end = hand.getLandmark(connection[1]);

  if (start != null && end != null) {
    canvas.drawLine(
      Offset(start.x, start.y),
      Offset(end.x, end.y),
      paint,
    );
  }
}
```

## Handedness

The `handedness` property indicates whether the detected hand is a left or right hand:

```dart
final Hand hand = hands.first;

if (hand.handedness == Handedness.left) {
  print('Left hand detected');
} else if (hand.handedness == Handedness.right) {
  print('Right hand detected');
}
```

## Gesture Recognition

Enable gesture recognition to classify hand poses into 7 gestures:

<!--
  Size these with width= (NOT height=). pub.dev's README stylesheet forces
  img{height:auto}, discarding height=, so height-sized side-by-side images
  render at natural width and stack. Percentage width= is honored by both
  pub.dev and GitHub; 24% + 61% is the pair's aspect-matched ratio and fits
  pub.dev's ~620px README column.
-->
<p align="center">
  <img src="assets/screenshots/gesture-thumbs-up.webp" alt="Thumbs-up gesture detection" width="24%">
  &nbsp;
  <img src="assets/screenshots/gesture-victory.webp" alt="Closed-fist and victory gesture detection" width="61%">
</p>

| Gesture | Description |
|---------|-------------|
| closedFist | Closed fist |
| openPalm | Open palm |
| pointingUp | Index finger pointing up |
| thumbDown | Thumbs down |
| thumbUp | Thumbs up |
| victory | Victory / peace sign |
| iLoveYou | "I love you" sign |

### Enabling Gestures

```dart
final detector = HandDetector(
  enableGestures: true,
  gestureMinConfidence: 0.5, // optional, default 0.5
);
await detector.initialize();

final hands = await detector.detect(imageBytes);
for (final hand in hands) {
  if (hand.hasGesture) {
    print('Gesture: ${hand.gesture!.type.name}');
    print('Confidence: ${hand.gesture!.confidence}');
  }
}
```

Gesture recognition uses a two-stage pipeline (gesture embedder + classifier) and requires `HandMode.boxesAndLandmarks` (the default mode).

## Detection Modes

This package supports two detection modes:

| Mode | Features | Speed |
|------|----------|-------|
| **boxesAndLandmarks** (default) | Bounding boxes + 21 landmarks + handedness | Standard |
| **boxes** | Bounding boxes only | Faster |

### Code Examples

```dart
// Full mode (default): bounding boxes + 21 landmarks + handedness
final detector = HandDetector(
  mode: HandMode.boxesAndLandmarks,
);

// Fast mode: bounding boxes only
final detector = HandDetector(
  mode: HandMode.boxes,
);
```

## Configuration Options

`HandDetector.create` (and the equivalent `initialize`) accept several configuration options:

```dart
final detector = await HandDetector.create(
  mode: HandMode.boxesAndLandmarks,       // Detection mode
  landmarkModel: HandLandmarkModel.full,  // Landmark model variant
  detectorConf: 0.45,                     // Palm detection confidence (0.0-1.0)
  palmNmsIou: 0.45,                       // Palm NMS IoU threshold (0.0-1.0)
  palmRoiScale: 2.6,                      // Palm ROI expansion fed to the landmark model
  maxDetections: 10,                      // Maximum hands to detect
  minLandmarkScore: 0.5,                  // Minimum landmark confidence (0.0-1.0)
  enableTracking: false,                  // MediaPipe-style cross-frame tracking
  trackingConfig: const TrackingConfig(), // ROI tuning used when tracking is on
  interpreterPoolSize: 1,                 // TFLite interpreter pool size
  performanceConfig: const PerformanceConfig(), // Performance config (default: auto)
  enableGestures: false,                  // Enable gesture recognition
  gestureMinConfidence: 0.5,              // Minimum gesture confidence (0.0-1.0)
);
```

### Detection tuning

These control the two model stages directly. The defaults match MediaPipe's shipped hand pipeline, so change them only if you need different behavior:

| Option | Default | Effect |
| --- | --- | --- |
| `detectorConf` | 0.45 | Palm detector score threshold. Raise to reject low-confidence palms, lower to catch more hands. |
| `palmNmsIou` | 0.45 | IoU threshold for palm non-maximum suppression. Higher keeps more overlapping detections; lower merges them harder. |
| `palmRoiScale` | 2.6 | How much the palm box is expanded before it is cropped for the landmark model. Larger includes more context around the palm (can help large/rotated hands); smaller crops tighter. |
| `minLandmarkScore` | 0.5 | Landmark-stage confidence gate. Hands whose landmark score falls below this are dropped. |
| `maxDetections` | 10 | Maximum number of hands returned per frame. |

### Tracking (`enableTracking` + `TrackingConfig`)

With `enableTracking: true`, the detector follows each hand frame-to-frame using a landmark-derived region of interest, and only runs the palm detector to discover new hands. This greatly reduces overlay drop-outs on video. Call `resetTracking()` between unrelated inputs (a new video, or independent still images) so a stale ROI is not reused.

`TrackingConfig` tunes the tracked ROI. It only takes effect when `enableTracking` is true, and its defaults port MediaPipe's hand tracking graph:

```dart
final detector = await HandDetector.create(
  enableTracking: true,
  trackingConfig: const TrackingConfig(
    roiScale: 2.0,        // ROI expansion for inter-frame motion margin
    roiShiftY: -0.1,      // shift toward the fingertips (negative = toward tips)
    associationIou: 0.5,  // IoU above which a fresh palm counts as an already-tracked hand
    minRoiSize: 0.03,     // drop tracking below this ROI size (normalized to the long image side)
    maxRoiSize: 1.2,      // drop tracking above this ROI size (normalized to the long image side)
  ),
);
```

> Tracking is implemented on native platforms. On web, `enableTracking` and `trackingConfig` are accepted for API parity but currently ignored (palm detection runs every frame).

## Live Camera Detection

For real-time hand detection from a camera feed, use `detectFromCameraImage`. All processing runs off the UI thread.

> **Desktop (Windows / macOS / Linux):** The default `camera` package does not include a streaming implementation for desktop platforms. You must also add [`camera_desktop`](https://pub.dev/packages/camera_desktop) to your `pubspec.yaml`, otherwise `startImageStream` throws `UnimplementedError: onStreamedFrameAvailable() is not implemented`.
> ```yaml
> dependencies:
>   camera: ^0.12.0
>   camera_desktop: ^1.2.0   # required for Windows, macOS, and Linux streaming
> ```

```dart
import 'package:camera/camera.dart';
import 'package:hand_detection/hand_detection.dart';

final detector = await HandDetector.create();

final cameras = await availableCameras();
final camera = CameraController(
  cameras.first,
  ResolutionPreset.medium,
  enableAudio: false,
  imageFormatGroup: ImageFormatGroup.yuv420, // prevents JPEG fallback on Android; ignored on desktop
);
await camera.initialize();

camera.startImageStream((CameraImage image) async {
  final hands = await detector.detectFromCameraImage(
    image,
    // rotation: rotationForFrame(...), // recommended on Android/iOS
    maxDim: 640,
  );
  // Process hands...
});
```

Tips:
- Pass `rotation:` on Android/iOS so the detector sees upright frames. Use `rotationForFrame(...)` to compute the correct value from sensor orientation and device orientation. On desktop frames are always upright so omit it.
- Pass `maxDim: 640` to downscale frames before inference. Recommended: full-res frames waste bandwidth since the model input is much smaller.
- Mirror the overlay on the front camera to match `CameraPreview`'s auto-mirrored texture.
- For advanced use, `prepareCameraFrame(...)` + `detectFromCameraFrame(...)` is the lower-level two-step API.

See the full [example app](https://pub.dev/packages/hand_detection/example) for a complete implementation.

## Background Processing

All inference runs automatically in a background isolate: the UI thread is never blocked during detection or gesture recognition. No special configuration is needed; `HandDetector` handles isolate management internally.

## Example

The [sample code](https://pub.dev/packages/hand_detection/example) from the pub.dev example tab includes a
Flutter app that paints detections onto an image: bounding boxes and 21-point hand landmarks with skeleton connections.
