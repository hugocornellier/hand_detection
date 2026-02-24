// Comprehensive unit tests for HandDetector.
//
// These tests cover:
// - Initialization and disposal
// - Error handling (works in standard test environment)
// - Detection with real sample images (requires device/platform-specific testing)
// - detect() and detectOnMat() methods
// - Different modes (boxes, boxesAndLandmarks)
// - Gesture recognition
// - Landmark and bounding box access
// - Configuration parameters
// - Edge cases
//
// NOTE: Most tests require TensorFlow Lite native libraries which are not
// available in the standard `flutter test` environment. To run all tests:
//
// - macOS: flutter test --platform=macos test/hand_detector_test.dart
// - Device: Run as integration tests on a physical device or emulator
//
// Tests that work in standard environment (no TFLite required):
// - StateError when not initialized
// - Parameter validation
//

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection_tflite/hand_detection_tflite.dart';
import 'test_config.dart';

// Sample images available in assets/samples/
const _singleHandA = 'assets/samples/istockphoto-462908027-612x612.jpg';
const _singleHandB =
    'assets/samples/360_F_554788951_fLAy5C8e9bha4caBTWVJN6rvTD0pEVfE.jpg';
const _twoHandsStanding = 'assets/samples/img-standing.png';
const _twoHandsSideBySide = 'assets/samples/2-hands.png';
const _twoPalms = 'assets/samples/two-palms.png';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('HandDetector - Initialization and Disposal', () {
    test('should initialize successfully with default options', () async {
      final detector = HandDetector();
      expect(detector.isInitialized, false);

      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
      expect(detector.isInitialized, false);
    });

    test('should initialize with custom configuration', () async {
      final detector = HandDetector(
        mode: HandMode.boxes,
        landmarkModel: HandLandmarkModel.full,
        detectorConf: 0.6,
        maxDetections: 5,
        minLandmarkScore: 0.7,
      );

      await detector.initialize();
      expect(detector.isInitialized, true);
      expect(detector.mode, HandMode.boxes);
      expect(detector.landmarkModel, HandLandmarkModel.full);
      expect(detector.detectorConf, 0.6);
      expect(detector.maxDetections, 5);
      expect(detector.minLandmarkScore, 0.7);

      await detector.dispose();
    });

    test('should allow re-initialization', () async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();
      expect(detector.isInitialized, true);

      // Re-initialize should work
      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
    });

    test('should handle multiple dispose calls', () async {
      final detector = HandDetector();
      await detector.initialize();
      await detector.dispose();
      expect(detector.isInitialized, false);

      // Second dispose should not throw
      await detector.dispose();
      expect(detector.isInitialized, false);
    });
  });

  group('HandDetector - Error Handling', () {
    test('should throw StateError when detect() called before initialize',
        () async {
      final detector = HandDetector();
      final bytes = TestUtils.createDummyImageBytes();

      expect(
        () => detector.detect(bytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    test('should throw StateError when detectOnMat() called before initialize',
        () async {
      final detector = HandDetector();
      final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);

      try {
        expect(
          () => detector.detectOnMat(mat),
          throwsA(isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('not initialized'),
          )),
        );
      } finally {
        mat.dispose();
      }
    });

    test('should return empty list for invalid image bytes', () async {
      final detector = HandDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      final results = await detector.detect(invalidBytes);

      expect(results, isEmpty);
      await detector.dispose();
    });
  });

  group('HandDetector - detect() with real images', () {
    test('should detect one hand in single-hand image with boxesAndLandmarks',
        () async {
      final detector = HandDetector(
        mode: HandMode.boxesAndLandmarks,
        landmarkModel: HandLandmarkModel.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results.length, 1);

      final hand = results.first;

      // Verify bounding box
      expect(hand.boundingBox, isNotNull);
      expect(hand.boundingBox.left, greaterThanOrEqualTo(0));
      expect(hand.boundingBox.top, greaterThanOrEqualTo(0));
      expect(hand.boundingBox.right, greaterThan(hand.boundingBox.left));
      expect(hand.boundingBox.bottom, greaterThan(hand.boundingBox.top));

      // Verify score
      expect(hand.score, greaterThan(0));
      expect(hand.score, lessThanOrEqualTo(1.0));

      // Verify landmarks (21 for MediaPipe hand model)
      expect(hand.hasLandmarks, true);
      expect(hand.landmarks.length, 21);

      // Check image dimensions
      expect(hand.imageWidth, greaterThan(0));
      expect(hand.imageHeight, greaterThan(0));

      await detector.dispose();
    });

    test('should detect one hand in second single-hand image', () async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandB);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results.length, 1);
      await detector.dispose();
    });

    test('should detect two hands in two-hands image', () async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load(_twoHandsSideBySide);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results.length, 2);
      await detector.dispose();
    });

    test('should detect hands with boxes-only mode', () async {
      final detector = HandDetector(
        mode: HandMode.boxes,
        landmarkModel: HandLandmarkModel.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      for (final hand in results) {
        // Should have bounding box
        expect(hand.boundingBox, isNotNull);
        expect(hand.score, greaterThan(0));

        // Should NOT have landmarks in boxes-only mode
        expect(hand.hasLandmarks, false);
        expect(hand.landmarks, isEmpty);
      }

      await detector.dispose();
    });
  });

  group('HandDetector - Gesture Recognition', () {
    test('should recognize gestures when enabled', () async {
      final detector = HandDetector(
        mode: HandMode.boxesAndLandmarks,
        landmarkModel: HandLandmarkModel.full,
        enableGestures: true,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      final hand = results.first;
      expect(hand.gesture, isNotNull);
      expect(hand.gesture!.type, GestureType.openPalm);
      expect(hand.gesture!.confidence, greaterThan(0.5));

      await detector.dispose();
    });

    test('should not have gesture when disabled', () async {
      final detector = HandDetector(
        mode: HandMode.boxesAndLandmarks,
        landmarkModel: HandLandmarkModel.full,
        enableGestures: false,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.gesture, isNull);

      await detector.dispose();
    });

    test('should detect handedness correctly', () async {
      final detector = HandDetector(
        mode: HandMode.boxesAndLandmarks,
        landmarkModel: HandLandmarkModel.full,
      );
      await detector.initialize();

      // Single hand images should detect right hand
      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results.length, 1);
      expect(results.first.handedness, Handedness.right);

      await detector.dispose();
    });
  });

  group('HandDetector - detectOnMat() method', () {
    test('should work with pre-decoded cv.Mat image', () async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      // Load and decode image using OpenCV
      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      expect(mat.isEmpty, false);

      try {
        // Use detectOnMat instead of detect
        final List<Hand> results = await detector.detectOnMat(mat);

        expect(results, isNotEmpty);

        for (final hand in results) {
          expect(hand.boundingBox, isNotNull);
          expect(hand.hasLandmarks, true);
          expect(hand.landmarks.length, 21);

          // Verify image dimensions match the decoded image
          expect(hand.imageWidth, mat.cols);
          expect(hand.imageHeight, mat.rows);
        }
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    test('detectOnMat() should give same results as detect()', () async {
      final detector = HandDetector(
        landmarkModel: HandLandmarkModel.full,
        detectorConf: 0.5,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();

      // Test with detect()
      final List<Hand> results1 = await detector.detect(bytes);

      // Test with detectOnMat()
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      try {
        final List<Hand> results2 = await detector.detectOnMat(mat);

        // Should detect same number of hands
        expect(results1.length, results2.length);

        // Scores should be identical (or very close)
        for (int i = 0; i < results1.length; i++) {
          expect(
            (results1[i].score - results2[i].score).abs(),
            lessThan(0.01),
          );
        }
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });
  });

  group('HandDetector - Landmark and BoundingBox Access', () {
    late HandDetector detector;
    late List<Hand> hands;

    setUpAll(() async {
      detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      hands = await detector.detect(bytes);
    });

    tearDownAll(() async {
      await detector.dispose();
    });

    test('should access specific landmarks by type', () {
      expect(hands, isNotEmpty);
      final hand = hands.first;

      // Test accessing different landmark types
      final wrist = hand.getLandmark(HandLandmarkType.wrist);
      expect(wrist, isNotNull);
      expect(wrist!.type, HandLandmarkType.wrist);

      final indexTip = hand.getLandmark(HandLandmarkType.indexFingerTip);
      expect(indexTip, isNotNull);
      expect(indexTip!.type, HandLandmarkType.indexFingerTip);

      final thumbTip = hand.getLandmark(HandLandmarkType.thumbTip);
      expect(thumbTip, isNotNull);
      expect(thumbTip!.type, HandLandmarkType.thumbTip);
    });

    test('should have valid landmark coordinates', () {
      final hand = hands.first;

      for (final landmark in hand.landmarks) {
        // Coordinates should be within image bounds
        expect(landmark.x, greaterThanOrEqualTo(0));
        expect(landmark.x, lessThanOrEqualTo(hand.imageWidth.toDouble()));
        expect(landmark.y, greaterThanOrEqualTo(0));
        expect(landmark.y, lessThanOrEqualTo(hand.imageHeight.toDouble()));

        // Visibility should be 0-1
        expect(landmark.visibility, greaterThanOrEqualTo(0));
        expect(landmark.visibility, lessThanOrEqualTo(1.0));

        // Z coordinate exists
        expect(landmark.z, isNotNull);
      }
    });

    test('should calculate normalized coordinates correctly', () {
      final hand = hands.first;
      final landmark = hand.landmarks.first;

      final xNorm = landmark.xNorm(hand.imageWidth);
      final yNorm = landmark.yNorm(hand.imageHeight);

      expect(xNorm, greaterThanOrEqualTo(0));
      expect(xNorm, lessThanOrEqualTo(1.0));
      expect(yNorm, greaterThanOrEqualTo(0));
      expect(yNorm, lessThanOrEqualTo(1.0));

      // Verify calculation
      expect(
        (xNorm - landmark.x / hand.imageWidth).abs(),
        lessThan(0.0001),
      );
      expect(
        (yNorm - landmark.y / hand.imageHeight).abs(),
        lessThan(0.0001),
      );
    });

    test('should convert landmark to pixel Point', () {
      final hand = hands.first;
      final landmark = hand.landmarks.first;

      final point = landmark.toPixel(hand.imageWidth, hand.imageHeight);

      expect(point.x, equals(landmark.x.toInt()));
      expect(point.y, equals(landmark.y.toInt()));
    });

    test('should access bounding box properties', () {
      final hand = hands.first;
      final bbox = hand.boundingBox;

      expect(bbox.left, greaterThanOrEqualTo(0));
      expect(bbox.top, greaterThanOrEqualTo(0));
      expect(bbox.right, greaterThan(bbox.left));
      expect(bbox.bottom, greaterThan(bbox.top));

      // Bounding box should be within image
      expect(bbox.left, lessThanOrEqualTo(hand.imageWidth.toDouble()));
      expect(bbox.top, lessThanOrEqualTo(hand.imageHeight.toDouble()));
      expect(bbox.right, lessThanOrEqualTo(hand.imageWidth.toDouble()));
      expect(bbox.bottom, lessThanOrEqualTo(hand.imageHeight.toDouble()));
    });

    test('should have all 21 landmarks', () {
      final hand = hands.first;
      expect(hand.landmarks.length, 21);

      // Verify we can access all 21 hand landmark types
      final landmarkTypes = [
        HandLandmarkType.wrist, // 0
        HandLandmarkType.thumbCMC, // 1
        HandLandmarkType.thumbMCP, // 2
        HandLandmarkType.thumbIP, // 3
        HandLandmarkType.thumbTip, // 4
        HandLandmarkType.indexFingerMCP, // 5
        HandLandmarkType.indexFingerPIP, // 6
        HandLandmarkType.indexFingerDIP, // 7
        HandLandmarkType.indexFingerTip, // 8
        HandLandmarkType.middleFingerMCP, // 9
        HandLandmarkType.middleFingerPIP, // 10
        HandLandmarkType.middleFingerDIP, // 11
        HandLandmarkType.middleFingerTip, // 12
        HandLandmarkType.ringFingerMCP, // 13
        HandLandmarkType.ringFingerPIP, // 14
        HandLandmarkType.ringFingerDIP, // 15
        HandLandmarkType.ringFingerTip, // 16
        HandLandmarkType.pinkyMCP, // 17
        HandLandmarkType.pinkyPIP, // 18
        HandLandmarkType.pinkyDIP, // 19
        HandLandmarkType.pinkyTip, // 20
      ];

      expect(landmarkTypes.length, 21);

      for (final type in landmarkTypes) {
        final landmark = hand.getLandmark(type);
        expect(landmark, isNotNull, reason: 'Missing landmark: $type');
        expect(landmark!.type, type);
      }
    });
  });

  group('HandDetector - Configuration Parameters', () {
    test('should respect detectorConf threshold', () async {
      // High confidence threshold
      final strictDetector = HandDetector(
        landmarkModel: HandLandmarkModel.full,
        detectorConf: 0.9,
      );
      await strictDetector.initialize();

      // Low confidence threshold
      final lenientDetector = HandDetector(
        landmarkModel: HandLandmarkModel.full,
        detectorConf: 0.3,
      );
      await lenientDetector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();

      final strictResults = await strictDetector.detect(bytes);
      final lenientResults = await lenientDetector.detect(bytes);

      // Lenient should detect same or more hands
      expect(lenientResults.length, greaterThanOrEqualTo(strictResults.length));

      await strictDetector.dispose();
      await lenientDetector.dispose();
    });

    test('should respect maxDetections parameter', () async {
      final detector = HandDetector(
        landmarkModel: HandLandmarkModel.full,
        maxDetections: 1,
      );
      await detector.initialize();

      // Use an image with 2 hands to verify maxDetections caps at 1
      final ByteData data = await rootBundle.load(_twoHandsSideBySide);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      // Should not detect more than maxDetections
      expect(results.length, lessThanOrEqualTo(1));

      await detector.dispose();
    });

    test('should respect minLandmarkScore parameter', () async {
      final detector = HandDetector(
        landmarkModel: HandLandmarkModel.full,
        minLandmarkScore: 0.9, // Very high threshold
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      // With high landmark score threshold, might get fewer results
      // or results without landmarks
      if (results.isNotEmpty) {
        for (final hand in results) {
          if (hand.hasLandmarks) {
            // If landmarks exist, they passed the quality threshold
            expect(hand.landmarks.length, 21);
          }
        }
      }

      await detector.dispose();
    });
  });

  group('HandDetector - Multiple Images', () {
    test('should process multiple single-hand images sequentially', () async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final images = [_singleHandA, _singleHandB];

      for (final imagePath in images) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();
        final List<Hand> results = await detector.detect(bytes);

        expect(results, isNotEmpty, reason: 'Failed to detect in $imagePath');
      }

      await detector.dispose();
    });

    test('should handle images with different hand counts', () async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final images = [
        _singleHandA,
        _twoHandsSideBySide,
        _twoPalms,
        _twoHandsStanding,
      ];

      for (final imagePath in images) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();
        final List<Hand> results = await detector.detect(bytes);

        // Should work regardless of image size
        if (results.isNotEmpty) {
          for (final hand in results) {
            expect(hand.imageWidth, greaterThan(0));
            expect(hand.imageHeight, greaterThan(0));
          }
        }
      }

      await detector.dispose();
    });
  });

  group('HandDetector - Sample Expected Counts', () {
    late HandDetector detector;

    setUp(() async {
      detector = HandDetector(
        landmarkModel: HandLandmarkModel.full,
        mode: HandMode.boxesAndLandmarks,
        detectorConf: 0.5,
        maxDetections: 10,
        minLandmarkScore: 0.5,
      );
      await detector.initialize();
    });

    tearDown(() async {
      await detector.dispose();
    });

    test('sample images yield expected hand counts', () async {
      final expectedCounts = <String, int>{
        _singleHandA: 1,
        _singleHandB: 1,
        _twoHandsStanding: 2,
        _twoHandsSideBySide: 2,
        _twoPalms: 2,
      };

      for (final entry in expectedCounts.entries) {
        final data = await rootBundle.load(entry.key);
        final bytes = data.buffer.asUint8List();
        final results = await detector.detect(bytes);

        expect(
          results.length,
          entry.value,
          reason: 'Unexpected hand count for ${entry.key}',
        );
      }
    });
  });

  group('HandDetector - Edge Cases', () {
    test('should handle empty landmarks list in boxes mode', () async {
      final detector = HandDetector(mode: HandMode.boxes);
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      for (final hand in results) {
        expect(hand.landmarks, isEmpty);
        expect(hand.hasLandmarks, false);

        // getLandmark should return null for any type
        expect(hand.getLandmark(HandLandmarkType.wrist), isNull);
      }

      await detector.dispose();
    });

    test('should handle 1x1 image', () async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final bytes = TestUtils.createDummyImageBytes();
      final List<Hand> results = await detector.detect(bytes);

      // Should not crash, but probably won't detect anything
      expect(results, isNotNull);

      await detector.dispose();
    });

    test('Hand.toString() should not crash', () async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load(_singleHandA);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      final handString = results.first.toString();
      expect(handString, isNotEmpty);
      expect(handString, contains('Hand('));
      expect(handString, contains('score='));
      expect(handString, contains('landmarks='));

      await detector.dispose();
    });
  });
}
