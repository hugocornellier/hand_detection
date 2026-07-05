## 3.3.0

* Add MediaPipe-style detection + tracking: pass `enableTracking: true` to `HandDetector.create` / `initialize` / `initializeFromBuffers`. Each detected hand is followed frame-to-frame via a rotated region of interest derived from its own landmarks (wrist to middle-finger-MCP orientation, tight landmark box expanded 2x), and the palm detector only runs to acquire new hands or re-acquire lost ones. This removes the per-frame palm re-detection drop-outs on video and live camera (a sample origami clip went from 90% to 100% of frames with a hand at the same 0.5 thresholds). Off by default; existing behavior is unchanged.
* Add `HandDetector.resetTracking()` to clear the cross-frame tracking state between unrelated inputs (a new video, or independent still images). Safe to call when tracking is disabled; no-op on web.
* Tracking ROIs are gated by `minLandmarkScore` (keep it near 0.5 with tracking; permissive thresholds let garbage frames perpetuate) and guarded against degenerate or runaway regions, so a bad frame falls back to palm re-detection instead of drifting.
* Web: `enableTracking` is accepted for cross-platform API parity (the web implementation still runs palm detection every frame).
* Example app: the Live Camera and Video File screens expose a tracking toggle in their settings, off by default.
* Expose palm post-processing tuning on `HandDetector.create` / `initialize` / `initializeFromBuffers`: `palmNmsIou` (palm non-maximum-suppression IoU) and `palmRoiScale` (how much the palm box is expanded before it is cropped for the landmark model). Both were previously hardcoded (0.45 / 2.6); the defaults are unchanged.
* Add `TrackingConfig` to tune the cross-frame tracked ROI (`roiScale`, `roiShiftY`, `associationIou`, `minRoiSize`, `maxRoiSize`), passed via `trackingConfig:`. These were previously hardcoded constants; the defaults port MediaPipe's hand tracking graph and are unchanged. Only takes effect when `enableTracking` is true.
* Web: `palmNmsIou` / `palmRoiScale` are honored (they feed the shared palm post-processing), and the web palm detector now applies `detectorConf` (previously accepted but ignored). `trackingConfig` is accepted for API parity but ignored (web has no ROI tracking yet).

## 3.2.0

* Update flutter_litert -> 3.2.0
* Import native-only flutter_litert APIs via `package:flutter_litert/native.dart` so they resolve under static analysis (flutter_litert 3.2.0 moved `InterpreterPool`, `IsolateWorkerBase`, and `TensorFloat32Views` behind the native conditional export). No runtime or API change.
* Default the public entry's conditional export to the web implementation, gating native behind `dart.library.io`, restoring WASM compatibility (pub.dev WASM-ready). No behavior change on any platform.
* Add `package:hand_detection/hand_detection_native.dart`, a native-only entry point that re-exports the native implementation for code that runs only on native platforms.

## 3.1.2

* Update flutter_litert -> 3.1.1

## 3.1.1

* Update flutter_litert -> 2.8.3

## 3.1.0

* Update flutter_litert -> 2.8.0
* Complete Swift Package Manager migration: example apps build via SPM without CocoaPods

## 3.0.7

* Remove unused Darwin podspecs for Dart-only iOS/macOS plugin registration.

## 3.0.6

* Update flutter_litert -> 2.5.8

## 3.0.5

* Update flutter_litert -> 2.5.5

## 3.0.4

* Update flutter_litert to 2.5.3 and camera_desktop to 1.1.4

## 3.0.3

* Update flutter_litert -> 2.5.2

## 3.0.2

* Update flutter_litert -> 2.5.0

## 3.0.1

* Update flutter_litert -> 2.4.1

## 3.0.0

**Breaking:**
* `HandDetector` configuration moves from the constructor to `initialize()`. `HandDetector({mode: ..., landmarkModel: ..., ...})` → `HandDetector()` + `await detector.initialize(mode: ..., landmarkModel: ..., ...)`. Matches `FaceDetector`'s shape. `HandDetector.create({...})` continues to accept the same named params unchanged.
* `HandDetector.detect` now takes `Uint8List` instead of `List<int>`. Callers passing a plain `List<int>` must convert (`Uint8List.fromList(...)`); callers already passing `Uint8List` (including `File.readAsBytes()` and `camera` plugin bytes) are unaffected.
* `detect(...)` no longer swallows exceptions. Previously, malformed image bytes resolved to an empty list; now they surface as an exception. Genuine errors (`StateError`, isolate failures, dispose races) also propagate. Wrap `detect(...)` in a `try/catch` if your callsite depended on the previous silent-failure behavior.

* `HandDetector` now runs all TFLite inference in a dedicated background isolate automatically, keeping the UI thread free.
* Deprecate `HandDetectorIsolate`: use `HandDetector` directly. `HandDetectorIsolate` is kept as a thin wrapper for backward compatibility and will be removed in a future release.
* Add `HandDetector.create({...})` static factory for one-step construction and initialization (mirrors `FaceDetector.create`).
* Add `detectFromFilepath(String path)` convenience method.
* Add `detectFromMatBytes(Uint8List, {required int width, required int height, int matType})` fast path: transfers raw pixel bytes to the background isolate via zero-copy `TransferableTypedData`, avoiding `cv.Mat` construction on the calling thread.
* Rename `detectOnMat` to `detectFromMat` and `detectOnMatBytes` to `detectFromMatBytes` for naming parity with `face_detection_tflite`; old names kept as deprecated aliases.
* Expand `flutter_litert` re-exports through the `hand_detection` barrel to match `face_detection_tflite`: tensor helpers (`createNHWCTensor4D`, `fillNHWC4D`, `allocTensorShape`, `flattenDynamicTensor`), math helpers (`sigmoid`, `sigmoidClipped`, `clamp01`, `clip`), letterbox helpers (`computeLetterboxParams`, `LetterboxParams`), BGR→RGB byte helpers (`bgrBytesToRgbFloat32`, `bgrBytesToSignedFloat32`), and `PerformanceMode`. Consumers no longer need a direct `flutter_litert` import for these.
* Update example app to use `HandDetector.create()` instead of `HandDetectorIsolate.spawn()`.
* Rewrite README's Live Camera Detection section around the shared `packYuv420` + native `cv.cvtColor` pattern, and drop the "Background Isolate Detection" / "OpenCV Mat Support" sections that pointed users at the deprecated `HandDetectorIsolate`.

## 2.1.2

* Add public `HandDetector.modelVersion` and `HandDetector.modelVersionFor(...)` APIs for downstream cache invalidation.

## 2.1.1

* Fix iOS camera preview lifecycle in example

## 2.1.0

* Fix Android live camera in the example app:
  * Replace the per-pixel Dart YUV→BGR loop with `flutter_litert`'s shared `packYuv420` helper + native `cv.cvtColor`, matching `face_detection_tflite`.
  * `_rotationFlagForFrame` now handles all four device orientations (portrait up/down, landscape left/right) via a combined `sensorOrientation` + `DeviceOrientation` formula. Previously only one of the two landscape directions rendered correctly; the other was 180° off.
  * Mirror the detection overlay on Android front camera to match `CameraPreview`'s auto-mirrored preview texture.
* Align example app live-camera layout with `face_detection_tflite`: Material+Row top bar (replaces AppBar), flip-camera button, FPS + detection-time display, rotating top bar in landscape with safe-area padding, and a settings popup housing hand-specific controls (Max Hands slider, gesture toggle).
* Re-export `packYuv420`, `YuvPlane`, `YuvLayout`, and `PackedYuv` from `flutter_litert` through the `hand_detection` barrel.
* Update `flutter_litert` to `^2.2.0`.

## 2.0.9

* Update flutter_litert -> 2.1.0

## 2.0.8

* Update flutter_litert to 2.0.13

## 2.0.7

* Update flutter_litert -> 2.0.12

## 2.0.6

* Update flutter_litert 2.0.10 -> 2.0.11

## 2.0.5

* Update documentation

## 2.0.4

* Update flutter_litert 2.0.8 -> 2.0.10

## 2.0.3

* Enable auto hardware acceleration by default (XNNPACK on all native platforms, Metal GPU on iOS)
* Update flutter_litert 2.0.6 -> 2.0.8

## 2.0.2

* Update flutter_litert 2.0.5 -> 2.0.6

## 2.0.1
 
* Fix Xcode build warnings by declaring PrivacyInfo.xcprivacy as a resource bundle in iOS and macOS podspecs 

## 2.0.0

**Breaking:** `Point` now uses `double` coordinates. `BoundingBox.toMap()` format changed to corner-based.

* Use shared `Point` and `BoundingBox` from `flutter_litert` 2.0.0
* `toPixel()` now returns full-precision `double` coordinates (was truncating to `int`)
* Remove duplicate NMS implementation, use shared `nms()` from `flutter_litert`
* Refactor isolate worker to use `IsolateWorkerBase` from flutter_litert
* Simplify model classes (PalmDetector, HandLandmarkModel, GestureRecognizer)
* Remove integration tests from unit test suite
* Remove dead test helpers (`test_config.dart`)

## 1.0.3

* Update `camera_desktop` 1.0.1 -> 1.0.3

## 1.0.2

* Update `flutter_litert` -> 1.2.0
* Refactor to use `flutter_litert` shared utilities (`InterpreterFactory`, `InterpreterPool`, `PerformanceConfig`, `generateAnchors`)

## 1.0.1

* Update `opencv_dart` 2.1.0 -> 2.2.1
* Update `flutter_litert` 1.0.2 -> 1.0.3

## 1.0.0

First stable release of `hand_detection`

### Pipeline

* **Palm detection**, SSD model with rotation-aware bounding boxes
* **Hand landmarks**, 21-point 3D landmarks with visibility scores
* **Gesture recognition**, 7 gestures (fist, open palm, pointing up, thumbs down/up, victory, I love you)
* **Handedness**, Left/right classification

### Features

* Two modes: `HandMode.boxes` (bounding boxes only) and `HandMode.boxesAndLandmarks` (full landmarks)
* `HandDetectorIsolate` for background-thread inference with zero-copy transfer
* Direct `cv.Mat` input for live camera processing
* XNNPACK hardware acceleration with configurable thread count
* Configurable confidence thresholds and detection limits

### Platforms

* iOS, Android, macOS, Windows, Linux

## 0.0.4

* Update documentation

## 0.0.3

* Update `flutter_litert` to 1.0.1, `camera` to 0.12.0

## 0.0.2

* Update `flutter_litert` to 0.2.2

## 0.0.1

* Initial release
