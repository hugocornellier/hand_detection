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