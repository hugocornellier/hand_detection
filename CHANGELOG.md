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
