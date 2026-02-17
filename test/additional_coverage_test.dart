import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection_tflite/hand_detection_tflite.dart';
import 'package:hand_detection_tflite/src/image_utils.dart';
import 'package:hand_detection_tflite/src/palm_detector.dart';
import 'package:hand_detection_tflite/src/hand_landmark_model.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('HandDetector returns empty when cv.imdecode fails', () async {
    final detector = HandDetector();
    await detector.initialize();

    // Invalid bytes should fail to decode
    final List<Hand> results = await detector.detect(const <int>[1, 2, 3]);

    try {
      expect(results, isEmpty);
    } finally {
      await detector.dispose();
    }
  });

  test('Dart registration is callable', () {
    final instance = HandDetectionTfliteDart();
    expect(instance, isA<HandDetectionTfliteDart>());
    expect(() => HandDetectionTfliteDart.registerWith(), returnsNormally);
  });

  group('HandLandmarkModelRunner', () {
    test('reports pool configuration and guard rails', () async {
      final runner = HandLandmarkModelRunner(poolSize: 7);
      expect(runner.poolSize, 7);
      expect(runner.isInitialized, isFalse);

      final mat = cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3);
      try {
        expect(
          () => runner.run(mat),
          throwsA(isA<StateError>()),
        );
      } finally {
        mat.dispose();
      }
    });
  });

  group('PalmDetector anchor generation', () {
    test('generates correct number of anchors', () {
      final options = SSDAnchorOptions(
        numLayers: 4,
        minScale: 0.1484375,
        maxScale: 0.75,
        inputSizeHeight: 192,
        inputSizeWidth: 192,
        anchorOffsetX: 0.5,
        anchorOffsetY: 0.5,
        strides: [8, 16, 16, 16],
        aspectRatios: [1.0],
        reduceBoxesInLowestLayer: false,
        interpolatedScaleAspectRatio: 1.0,
        fixedAnchorSize: true,
      );

      final anchors = PalmDetector.generateAnchors(options);

      // Layer 0 (stride=8): 24x24 = 576 anchors × 2 = 1152
      // Layer 1 (stride=16): 12x12 = 144 anchors × 2 = 288
      // Layer 2 (stride=16): 12x12 = 144 anchors × 2 = 288
      // Layer 3 (stride=16): 12x12 = 144 anchors × 2 = 288
      // Total: 2016 anchors
      expect(anchors.length, 2016);

      // All anchors should have 4 values [cx, cy, w, h]
      for (final anchor in anchors) {
        expect(anchor.length, 4);
        // With fixed anchor size, w and h should be 1.0
        expect(anchor[2], 1.0);
        expect(anchor[3], 1.0);
      }
    });

    test('normalizeRadians normalizes angles correctly', () {
      // Test values within range
      expect(PalmDetector.normalizeRadians(0), closeTo(0, 0.0001));
      expect(PalmDetector.normalizeRadians(1.0), closeTo(1.0, 0.0001));
      expect(PalmDetector.normalizeRadians(-1.0), closeTo(-1.0, 0.0001));

      // Test values outside range
      const pi = 3.14159265359;
      expect(PalmDetector.normalizeRadians(4.0), closeTo(4.0 - 2 * pi, 0.01));
      expect(PalmDetector.normalizeRadians(-4.0), closeTo(-4.0 + 2 * pi, 0.01));
    });
  });

  group('ImageUtils rotation utilities', () {
    test('keepAspectResizeAndPad maintains aspect ratio', () {
      final source = cv.Mat.zeros(200, 100, cv.MatType.CV_8UC3);
      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(source, 192, 192);

      try {
        expect(padded.cols, 192);
        expect(padded.rows, 192);

        // Resized should maintain aspect ratio (100:200 = 1:2)
        // With target 192x192, height-constrained resize gives 96x192
        expect(resized.cols, 96);
        expect(resized.rows, 192);
      } finally {
        source.dispose();
        padded.dispose();
        resized.dispose();
      }
    });

    test('palmToRect converts normalized coordinates', () {
      final palm = PalmDetection(
        sqnRrSize: 0.5,
        rotation: 0.0,
        sqnRrCenterX: 0.5,
        sqnRrCenterY: 0.5,
        score: 0.9,
      );

      final rect = ImageUtils.palmToRect(palm, 640, 480);

      expect(rect[0], 320); // cx = 0.5 * 640
      expect(rect[1], 240); // cy = 0.5 * 480
      expect(rect[2], 320); // size = 0.5 * max(640, 480) = 0.5 * 640
      expect(rect[3], 320); // height = same as width
      expect(rect[4], closeTo(0.0, 0.0001)); // angle in degrees
    });

    test('letterbox224 produces 224x224 output', () {
      final src = cv.Mat.zeros(150, 100, cv.MatType.CV_8UC3);
      final ratioOut = <double>[];
      final dwdhOut = <int>[];

      final result = ImageUtils.letterbox224(src, ratioOut, dwdhOut);

      try {
        expect(result.cols, 224);
        expect(result.rows, 224);
        expect(ratioOut.isNotEmpty, true);
        expect(dwdhOut.length, 2);
      } finally {
        src.dispose();
        result.dispose();
      }
    });

    test('matToFloat32Tensor converts BGR to RGB', () {
      // Create a 2x2 blue image (BGR: 255, 0, 0)
      final mat = cv.Mat.zeros(2, 2, cv.MatType.CV_8UC3);
      mat.setTo(cv.Scalar(255, 0, 0, 0)); // Pure blue in BGR

      try {
        final tensor = ImageUtils.matToFloat32Tensor(mat);

        // First pixel should be RGB (0, 0, 1) after BGR->RGB conversion
        expect(tensor[0], closeTo(0.0, 0.01)); // R = 0
        expect(tensor[1], closeTo(0.0, 0.01)); // G = 0
        expect(
            tensor[2], closeTo(1.0, 0.01)); // B = 1 (was 255 in BGR B channel)
      } finally {
        mat.dispose();
      }
    });
  });

  group('Types', () {
    test('HandLandmarkType has 21 values', () {
      expect(HandLandmarkType.values.length, 21);
    });

    test('handLandmarkConnections has valid connections', () {
      // Should have 21 connections for the hand skeleton
      expect(handLandmarkConnections.length, 21);

      // All connections should reference valid landmark types
      for (final connection in handLandmarkConnections) {
        expect(connection.length, 2);
        expect(connection[0].index, lessThan(21));
        expect(connection[1].index, lessThan(21));
      }
    });

    test('numHandLandmarks constant is 21', () {
      expect(numHandLandmarks, 21);
    });

    test('Handedness enum has left and right', () {
      expect(Handedness.values.length, 2);
      expect(Handedness.left.index, 0);
      expect(Handedness.right.index, 1);
    });

    test('Hand can include handedness', () {
      final hand = Hand(
        boundingBox: BoundingBox(left: 0, top: 0, right: 100, bottom: 100),
        score: 0.9,
        landmarks: const [],
        imageWidth: 640,
        imageHeight: 480,
        handedness: Handedness.right,
      );

      expect(hand.handedness, Handedness.right);
    });

    test('HandLandmarks includes handedness', () {
      final landmarks = HandLandmarks(
        landmarks: const [],
        worldLandmarks: const [],
        score: 0.9,
        handedness: Handedness.left,
      );

      expect(landmarks.handedness, Handedness.left);
      expect(landmarks.score, 0.9);
    });
  });
}
