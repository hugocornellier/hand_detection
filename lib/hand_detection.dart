/// On-device hand detection and landmark estimation using TensorFlow Lite.
///
/// This library provides a Flutter plugin for real-time human hand detection
/// using MediaPipe-style models. It detects hands in images and extracts
/// 21 body landmarks (keypoints) for each detected hand.
///
/// This is a port of a Python hand detection library that uses:
/// - SSD-based palm detection with 2016 anchors
/// - MediaPipe hand landmark model (21 landmarks)
/// - Rotation-aware cropping for hand alignment
/// - MediaPipe gesture recognition (optional)
///
/// **Quick Start:**
/// ```dart
/// import 'package:hand_detection/hand_detection.dart';
///
/// final detector = HandDetector();
/// await detector.initialize();
///
/// final hands = await detector.detect(imageBytes);
/// for (final hand in hands) {
///   print('Hand detected at ${hand.boundingBox}');
///   print('Handedness: ${hand.handedness}');
///   if (hand.hasLandmarks) {
///     final wrist = hand.getLandmark(HandLandmarkType.wrist);
///     print('Wrist position: (${wrist?.x}, ${wrist?.y})');
///   }
/// }
///
/// await detector.dispose();
/// ```
///
/// **Gesture Recognition:**
/// ```dart
/// final detector = HandDetector(enableGestures: true);
/// await detector.initialize();
///
/// final hands = await detector.detect(imageBytes);
/// for (final hand in hands) {
///   if (hand.gesture != null) {
///     print('Gesture: ${hand.gesture!.type}'); // thumbUp, victory, etc.
///     print('Confidence: ${hand.gesture!.confidence}');
///   }
/// }
/// ```
///
/// **Main Classes:**
/// - [HandDetectorIsolate]: Background isolate wrapper for hand detection
/// - [HandDetector]: Main API for hand detection
/// - [Hand]: Detected hand with bounding box, landmarks, handedness, and gesture
/// - [HandLandmark]: Single keypoint with 3D coordinates (x, y, z) and visibility
/// - [HandLandmarkType]: Enum of 21 hand landmarks (wrist, finger joints, tips)
/// - [Handedness]: Left or right hand indication
/// - [BoundingBox]: Axis-aligned rectangle for hand location
/// - [GestureType]: Recognized gesture (thumbUp, victory, closedFist, etc.)
/// - [GestureResult]: Gesture type with confidence score
///
/// **Detection Modes:**
/// - [HandMode.boxes]: Fast detection returning only bounding boxes
/// - [HandMode.boxesAndLandmarks]: Full pipeline with 21 landmarks per hand
///
/// **Supported Gestures (when enableGestures: true):**
/// - closedFist, openPalm, pointingUp, thumbDown, thumbUp, victory, iLoveYou
///
/// **Model Variant:**
/// - [HandLandmarkModel.full]: Full model (only variant available)
///
/// **21 Hand Landmarks:**
/// - Wrist (0)
/// - Thumb: CMC (1), MCP (2), IP (3), Tip (4)
/// - Index: MCP (5), PIP (6), DIP (7), Tip (8)
/// - Middle: MCP (9), PIP (10), DIP (11), Tip (12)
/// - Ring: MCP (13), PIP (14), DIP (15), Tip (16)
/// - Pinky: MCP (17), PIP (18), DIP (19), Tip (20)
library;

export 'src/types.dart';
export 'src/hand_detector.dart' show HandDetector;
export 'src/isolate/hand_detector_isolate.dart' show HandDetectorIsolate;
export 'src/models/palm_detector.dart' show PalmDetection;
export 'src/dart_registration.dart';

// Re-export cv.Mat for users who want to use detectOnMat directly
export 'package:opencv_dart/opencv_dart.dart' show Mat, imdecode, IMREAD_COLOR;
