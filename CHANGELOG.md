## 1.0.0

First stable release of `hand_detection`

### Pipeline

* **Palm detection** — SSD model with rotation-aware bounding boxes
* **Hand landmarks** — 21-point 3D landmarks with visibility scores
* **Gesture recognition** — 7 gestures (fist, open palm, pointing up, thumbs down/up, victory, I love you)
* **Handedness** — Left/right classification

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