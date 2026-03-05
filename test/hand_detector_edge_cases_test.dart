import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/hand_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('HandDetector constructor', () {
    test('default values', () {
      final detector = HandDetector();
      expect(detector.mode, HandMode.boxesAndLandmarks);
      expect(detector.landmarkModel, HandLandmarkModel.full);
      expect(detector.detectorConf, 0.45);
      expect(detector.maxDetections, 10);
      expect(detector.minLandmarkScore, 0.5);
      expect(detector.interpreterPoolSize, 1);
      expect(detector.performanceConfig.mode, PerformanceMode.disabled);
      expect(detector.enableGestures, false);
      expect(detector.gestureMinConfidence, 0.5);
      expect(detector.isInitialized, false);
    });

    test('custom interpreterPoolSize preserved when no perf config', () {
      final detector = HandDetector(interpreterPoolSize: 3);
      expect(detector.interpreterPoolSize, 3);
    });

    test('interpreterPoolSize forced to 1 when performance config enabled', () {
      final detector = HandDetector(
        interpreterPoolSize: 4,
        performanceConfig: const PerformanceConfig.xnnpack(numThreads: 2),
      );
      // With XNNPACK delegate, pool size is forced to 1
      expect(detector.interpreterPoolSize, 1);
    });

    test('interpreterPoolSize preserved when performance disabled', () {
      final detector = HandDetector(
        interpreterPoolSize: 4,
        performanceConfig: PerformanceConfig.disabled,
      );
      expect(detector.interpreterPoolSize, 4);
    });
  });

  group('HandDetector initializeFromBuffers', () {
    test('initializes correctly with model bytes', () async {
      final palmBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/hand_detection.tflite')
            .readAsBytesSync(),
      );
      final landmarkBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/hand_landmark_full.tflite')
            .readAsBytesSync(),
      );

      final detector = HandDetector();
      await detector.initializeFromBuffers(
        palmDetectionBytes: palmBytes,
        handLandmarkBytes: landmarkBytes,
      );

      expect(detector.isInitialized, true);
      await detector.dispose();
    });

    test('initializes with gesture models', () async {
      final palmBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/hand_detection.tflite')
            .readAsBytesSync(),
      );
      final landmarkBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/hand_landmark_full.tflite')
            .readAsBytesSync(),
      );
      final embedderBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/gesture_embedder.tflite')
            .readAsBytesSync(),
      );
      final classifierBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/canned_gesture_classifier.tflite')
            .readAsBytesSync(),
      );

      final detector = HandDetector(enableGestures: true);
      await detector.initializeFromBuffers(
        palmDetectionBytes: palmBytes,
        handLandmarkBytes: landmarkBytes,
        gestureEmbedderBytes: embedderBytes,
        gestureClassifierBytes: classifierBytes,
      );

      expect(detector.isInitialized, true);

      // Detect on an image to verify full pipeline works
      final imageBytes = File(
        '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg',
      ).readAsBytesSync();
      final results = await detector.detect(imageBytes);

      expect(results, isNotEmpty);
      expect(results.first.gesture, isNotNull);

      await detector.dispose();
    });

    test('re-initialization via initializeFromBuffers', () async {
      final palmBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/hand_detection.tflite')
            .readAsBytesSync(),
      );
      final landmarkBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/hand_landmark_full.tflite')
            .readAsBytesSync(),
      );

      final detector = HandDetector();
      await detector.initializeFromBuffers(
        palmDetectionBytes: palmBytes,
        handLandmarkBytes: landmarkBytes,
      );
      expect(detector.isInitialized, true);

      // Re-init
      await detector.initializeFromBuffers(
        palmDetectionBytes: palmBytes,
        handLandmarkBytes: landmarkBytes,
      );
      expect(detector.isInitialized, true);

      await detector.dispose();
    });
  });

  group('HandDetector detectOnMat edge cases', () {
    late HandDetector detector;

    setUpAll(() async {
      detector = HandDetector(landmarkModel: HandLandmarkModel.full);
      await detector.initialize();
    });

    tearDownAll(() async {
      await detector.dispose();
    });

    test('very small image returns empty or valid results', () async {
      final mat = cv.Mat.zeros(5, 5, cv.MatType.CV_8UC3);
      try {
        final results = await detector.detectOnMat(mat);
        expect(results, isNotNull);
        // May be empty, that's fine
      } finally {
        mat.dispose();
      }
    });

    test('large image processes without error', () async {
      final mat = cv.Mat.zeros(1080, 1920, cv.MatType.CV_8UC3);
      try {
        final results = await detector.detectOnMat(mat);
        expect(results, isNotNull);
      } finally {
        mat.dispose();
      }
    });

    test('sequential detect calls on different images', () async {
      final imagePathA =
          '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg';
      final imagePathB = '${Directory.current.path}/assets/samples/2-hands.png';

      final matA = cv.imdecode(
        Uint8List.fromList(File(imagePathA).readAsBytesSync()),
        cv.IMREAD_COLOR,
      );
      final matB = cv.imdecode(
        Uint8List.fromList(File(imagePathB).readAsBytesSync()),
        cv.IMREAD_COLOR,
      );

      try {
        final resultsA = await detector.detectOnMat(matA);
        final resultsB = await detector.detectOnMat(matB);

        expect(resultsA.length, 1);
        expect(resultsB.length, 2);
      } finally {
        matA.dispose();
        matB.dispose();
      }
    });
  });

  group('HandDetector boxes mode details', () {
    test('boxes mode includes rotation data', () async {
      final detector = HandDetector(mode: HandMode.boxes);
      await detector.initialize();

      final ByteData data = await rootBundle.load(
        'assets/samples/istockphoto-462908027-612x612.jpg',
      );
      final Uint8List bytes = data.buffer.asUint8List();
      final results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      final hand = results.first;

      // In boxes mode, rotation data should be present
      expect(hand.rotation, isNotNull);
      expect(hand.rotatedCenterX, isNotNull);
      expect(hand.rotatedCenterY, isNotNull);
      expect(hand.rotatedSize, isNotNull);

      // But no landmarks
      expect(hand.landmarks, isEmpty);
      expect(hand.hasLandmarks, false);

      // No handedness in boxes-only mode (comes from landmark model)
      expect(hand.handedness, isNull);

      await detector.dispose();
    });
  });

  group('HandDetector detect() error paths', () {
    test('empty byte list returns empty', () async {
      final detector = HandDetector();
      await detector.initialize();

      final results = await detector.detect([]);
      expect(results, isEmpty);

      await detector.dispose();
    });

    test('truncated PNG header returns empty', () async {
      final detector = HandDetector();
      await detector.initialize();

      // Valid PNG signature but truncated
      final results = await detector.detect([0x89, 0x50, 0x4E, 0x47]);
      expect(results, isEmpty);

      await detector.dispose();
    });

    test('random bytes return empty', () async {
      final detector = HandDetector();
      await detector.initialize();

      final results = await detector.detect(
        List.generate(100, (i) => i % 256),
      );
      expect(results, isEmpty);

      await detector.dispose();
    });
  });

  group('HandDetector with gestures enabled', () {
    test('gesture confidence respects gestureMinConfidence', () async {
      final detector = HandDetector(
        enableGestures: true,
        gestureMinConfidence: 0.1, // Very low threshold
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(
        'assets/samples/istockphoto-462908027-612x612.jpg',
      );
      final Uint8List bytes = data.buffer.asUint8List();
      final results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      final hand = results.first;

      // With low threshold, should always get a gesture
      expect(hand.gesture, isNotNull);

      await detector.dispose();
    });

    test('gestures not initialized when disabled', () async {
      final detector = HandDetector(enableGestures: false);
      await detector.initialize();

      final ByteData data = await rootBundle.load(
        'assets/samples/istockphoto-462908027-612x612.jpg',
      );
      final Uint8List bytes = data.buffer.asUint8List();
      final results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.gesture, isNull);
      expect(results.first.hasGesture, false);

      await detector.dispose();
    });
  });

  group('HandDetector initializeFromBuffers without gesture bytes', () {
    test('gesture recognizer not initialized when bytes are null', () async {
      final palmBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/hand_detection.tflite')
            .readAsBytesSync(),
      );
      final landmarkBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/hand_landmark_full.tflite')
            .readAsBytesSync(),
      );

      final detector = HandDetector(enableGestures: true);
      await detector.initializeFromBuffers(
        palmDetectionBytes: palmBytes,
        handLandmarkBytes: landmarkBytes,
        // No gesture model bytes
      );

      expect(detector.isInitialized, true);

      final imageBytes = File(
        '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg',
      ).readAsBytesSync();
      final results = await detector.detect(imageBytes);

      // Gestures should be null because gesture models weren't loaded
      for (final hand in results) {
        expect(hand.gesture, isNull);
      }

      await detector.dispose();
    });
  });

  group('HandDetector with multiple pool sizes', () {
    test('pool size 1 detects correctly', () async {
      final detector = HandDetector(interpreterPoolSize: 1);
      await detector.initialize();

      final ByteData data = await rootBundle.load(
        'assets/samples/istockphoto-462908027-612x612.jpg',
      );
      final results = await detector.detect(data.buffer.asUint8List());

      expect(results.length, 1);
      expect(results.first.landmarks.length, 21);

      await detector.dispose();
    });

    test('pool size 2 detects correctly', () async {
      final detector = HandDetector(interpreterPoolSize: 2);
      await detector.initialize();

      final ByteData data = await rootBundle.load(
        'assets/samples/2-hands.png',
      );
      final results = await detector.detect(data.buffer.asUint8List());

      expect(results.length, 2);
      for (final hand in results) {
        expect(hand.landmarks.length, 21);
      }

      await detector.dispose();
    });
  });

  group('HandDetector portrait image handling', () {
    test('detects hands in portrait-oriented image', () async {
      final detector = HandDetector();
      await detector.initialize();

      // Load square image and crop to portrait
      final ByteData data = await rootBundle.load(
        'assets/samples/istockphoto-462908027-612x612.jpg',
      );
      final bytes = data.buffer.asUint8List();
      final original = cv.imdecode(bytes, cv.IMREAD_COLOR);
      final portrait = original.region(cv.Rect(100, 0, 300, original.rows));

      try {
        expect(portrait.rows, greaterThan(portrait.cols));
        final results = await detector.detectOnMat(portrait);
        // Should not crash with portrait orientation
        expect(results, isNotNull);
      } finally {
        portrait.dispose();
        original.dispose();
        await detector.dispose();
      }
    });
  });

  group('HandDetector maxDetections behavior', () {
    test('maxDetections limits results from detectOnMat', () async {
      final detector = HandDetector(maxDetections: 1);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/2-hands.png');
      final bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final results = await detector.detectOnMat(mat);
        expect(results.length, lessThanOrEqualTo(1));
      } finally {
        mat.dispose();
        await detector.dispose();
      }
    });
  });

  group('HandDetector landmark transformations', () {
    test('landmarks are within image bounds after transformation', () async {
      final detector = HandDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(
        'assets/samples/istockphoto-462908027-612x612.jpg',
      );
      final results = await detector.detect(data.buffer.asUint8List());

      expect(results, isNotEmpty);
      final hand = results.first;

      for (final lm in hand.landmarks) {
        expect(lm.x, greaterThanOrEqualTo(0));
        expect(lm.x, lessThanOrEqualTo(hand.imageWidth.toDouble()));
        expect(lm.y, greaterThanOrEqualTo(0));
        expect(lm.y, lessThanOrEqualTo(hand.imageHeight.toDouble()));
      }

      await detector.dispose();
    });

    test('bounding box is within image bounds', () async {
      final detector = HandDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(
        'assets/samples/istockphoto-462908027-612x612.jpg',
      );
      final results = await detector.detect(data.buffer.asUint8List());

      expect(results, isNotEmpty);
      final hand = results.first;
      final bbox = hand.boundingBox;

      expect(bbox.left, greaterThanOrEqualTo(0));
      expect(bbox.top, greaterThanOrEqualTo(0));
      expect(bbox.right, lessThanOrEqualTo(hand.imageWidth.toDouble()));
      expect(bbox.bottom, lessThanOrEqualTo(hand.imageHeight.toDouble()));

      await detector.dispose();
    });
  });
}
