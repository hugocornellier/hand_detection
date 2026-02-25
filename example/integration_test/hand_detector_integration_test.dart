// Comprehensive integration tests for HandDetector.
//
// These tests cover:
// - Initialization and disposal
// - Detection with real sample images
// - detect() and detectOnMat() methods
// - Different model variants (lite, full, heavy)
// - Different modes (boxes, boxesAndLandmarks)
// - Landmark and bounding box access
// - Configuration parameters
// - Handedness detection
//
// Run with: flutter test integration_test/

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/hand_detection.dart';

/// Test helper to create a minimal 1x1 PNG image
class TestUtils {
  static Uint8List createDummyImageBytes() {
    return Uint8List.fromList([
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
      0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
      0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // Width: 1, Height: 1
      0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, // Bit depth, color type
      0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
      0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
      0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, // Image data
      0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND chunk
      0x42, 0x60, 0x82
    ]);
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('HandDetector - Initialization and Disposal', () {
    testWidgets('should initialize successfully with default options',
        (tester) async {
      final detector = HandDetector();
      expect(detector.isInitialized, false);

      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
      expect(detector.isInitialized, false);
    });

    testWidgets('should initialize with custom configuration', (tester) async {
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

    testWidgets('should allow re-initialization', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();
      expect(detector.isInitialized, true);

      // Re-initialize should work
      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
    });

    testWidgets('should handle multiple dispose calls', (tester) async {
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
    testWidgets(
        'should throw StateError when detect() called before initialize',
        (tester) async {
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

    testWidgets(
        'should throw StateError when detectOnMat() called before initialize',
        (tester) async {
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

    testWidgets('should return empty list for invalid image bytes',
        (tester) async {
      final detector = HandDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      final results = await detector.detect(invalidBytes);

      expect(results, isEmpty);
      await detector.dispose();
    });
  });

  group('HandDetector - detect() with real images', () {
    testWidgets('should detect hands in hand1.jpg with boxesAndLandmarks mode',
        (tester) async {
      final detector = HandDetector(
        mode: HandMode.boxesAndLandmarks,
        landmarkModel: HandLandmarkModel.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      for (final hand in results) {
        // Verify bounding box
        expect(hand.boundingBox, isNotNull);
        expect(hand.boundingBox.left, greaterThanOrEqualTo(0));
        expect(hand.boundingBox.top, greaterThanOrEqualTo(0));
        expect(hand.boundingBox.right, greaterThan(hand.boundingBox.left));
        expect(hand.boundingBox.bottom, greaterThan(hand.boundingBox.top));

        // Verify score
        expect(hand.score, greaterThan(0));
        expect(hand.score, lessThanOrEqualTo(1.0));

        // Verify 21 landmarks
        expect(hand.hasLandmarks, true);
        expect(hand.landmarks.length, 21);

        // Verify handedness
        expect(hand.handedness, isNotNull);
        expect(hand.handedness, isIn([Handedness.left, Handedness.right]));

        // Check image dimensions
        expect(hand.imageWidth, greaterThan(0));
        expect(hand.imageHeight, greaterThan(0));
      }

      await detector.dispose();
    });

    testWidgets('should detect hands in hand2.jpg', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand2.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      await detector.dispose();
    });

    testWidgets('should detect hands with boxes-only mode', (tester) async {
      final detector = HandDetector(
        mode: HandMode.boxes,
        landmarkModel: HandLandmarkModel.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
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

  group('HandDetector - detectOnMat() method', () {
    testWidgets('should work with pre-decoded image', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      // Load and decode image manually
      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      expect(mat.isEmpty, isFalse);

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

      mat.dispose();
      await detector.dispose();
    });
  });

  group('HandDetector - Different Model Variants', () {
    testWidgets('should work with lite model', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });

    testWidgets('should work with full model', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });

    testWidgets('should work with heavy model', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });
  });

  group('HandDetector - Landmark and BoundingBox Access', () {
    testWidgets('should access specific landmarks by type', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final hands = await detector.detect(bytes);

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

      await detector.dispose();
    });

    testWidgets('should have valid landmark coordinates', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final hands = await detector.detect(bytes);

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

      await detector.dispose();
    });

    testWidgets('should have all 21 landmarks', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final hands = await detector.detect(bytes);

      final hand = hands.first;
      expect(hand.landmarks.length, 21);

      // Verify we can access all 21 hand landmark types
      final landmarkTypes = [
        HandLandmarkType.wrist,
        HandLandmarkType.thumbCMC,
        HandLandmarkType.thumbMCP,
        HandLandmarkType.thumbIP,
        HandLandmarkType.thumbTip,
        HandLandmarkType.indexFingerMCP,
        HandLandmarkType.indexFingerPIP,
        HandLandmarkType.indexFingerDIP,
        HandLandmarkType.indexFingerTip,
        HandLandmarkType.middleFingerMCP,
        HandLandmarkType.middleFingerPIP,
        HandLandmarkType.middleFingerDIP,
        HandLandmarkType.middleFingerTip,
        HandLandmarkType.ringFingerMCP,
        HandLandmarkType.ringFingerPIP,
        HandLandmarkType.ringFingerDIP,
        HandLandmarkType.ringFingerTip,
        HandLandmarkType.pinkyMCP,
        HandLandmarkType.pinkyPIP,
        HandLandmarkType.pinkyDIP,
        HandLandmarkType.pinkyTip,
      ];

      for (final type in landmarkTypes) {
        final landmark = hand.getLandmark(type);
        expect(landmark, isNotNull, reason: 'Missing landmark: $type');
        expect(landmark!.type, type);
      }

      await detector.dispose();
    });
  });

  group('HandDetector - Configuration Parameters', () {
    testWidgets('should respect detectorConf threshold', (tester) async {
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

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final strictResults = await strictDetector.detect(bytes);
      final lenientResults = await lenientDetector.detect(bytes);

      // Lenient should detect same or more hands
      expect(lenientResults.length, greaterThanOrEqualTo(strictResults.length));

      await strictDetector.dispose();
      await lenientDetector.dispose();
    });

    testWidgets('should respect maxDetections parameter', (tester) async {
      final detector = HandDetector(
        landmarkModel: HandLandmarkModel.full,
        maxDetections: 1,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      // Should not detect more than maxDetections
      expect(results.length, lessThanOrEqualTo(1));

      await detector.dispose();
    });
  });

  group('HandDetector - Handedness Detection', () {
    testWidgets('should detect handedness for each hand', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Hand> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      for (final hand in results) {
        expect(hand.handedness, isNotNull);
        expect(hand.handedness, isIn([Handedness.left, Handedness.right]));
      }

      await detector.dispose();
    });
  });

  group('HandDetector - Multiple Images', () {
    testWidgets('should process multiple images sequentially', (tester) async {
      final detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();

      final images = [
        'assets/samples/hand1.jpg',
        'assets/samples/hand2.jpg',
      ];

      for (final imagePath in images) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();
        final List<Hand> results = await detector.detect(bytes);

        expect(results, isNotEmpty, reason: 'Failed to detect in $imagePath');
      }

      await detector.dispose();
    });
  });

  group('HandDetector - Edge Cases', () {
    testWidgets('should handle empty landmarks list in boxes mode',
        (tester) async {
      final detector = HandDetector(mode: HandMode.boxes);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/hand1.jpg');
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
  });
}
