import 'dart:math' as math;
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/src/models/palm_detector.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('SSDAnchorOptions', () {
    test('stores all configuration fields', () {
      const options = SSDAnchorOptions(
        numLayers: 4,
        minScale: 0.1,
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

      expect(options.numLayers, 4);
      expect(options.minScale, 0.1);
      expect(options.maxScale, 0.75);
      expect(options.inputSizeHeight, 192);
      expect(options.inputSizeWidth, 192);
      expect(options.anchorOffsetX, 0.5);
      expect(options.anchorOffsetY, 0.5);
      expect(options.strides, [8, 16, 16, 16]);
      expect(options.aspectRatios, [1.0]);
      expect(options.reduceBoxesInLowestLayer, false);
      expect(options.interpolatedScaleAspectRatio, 1.0);
      expect(options.fixedAnchorSize, true);
    });
  });

  group('PalmDetection', () {
    test('stores all fields', () {
      const detection = PalmDetection(
        sqnRrSize: 0.5,
        rotation: 0.3,
        sqnRrCenterX: 0.4,
        sqnRrCenterY: 0.6,
        score: 0.95,
      );

      expect(detection.sqnRrSize, 0.5);
      expect(detection.rotation, 0.3);
      expect(detection.sqnRrCenterX, 0.4);
      expect(detection.sqnRrCenterY, 0.6);
      expect(detection.score, 0.95);
    });
  });

  group('generateAnchors', () {
    test('generates 2016 anchors for palm detection config', () {
      const options = SSDAnchorOptions(
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

      final anchors = generateAnchors(options);
      expect(anchors.length, 2016);
    });

    test('all anchors have 4 values [cx, cy, w, h]', () {
      const options = SSDAnchorOptions(
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

      final anchors = generateAnchors(options);
      for (final anchor in anchors) {
        expect(anchor.length, 4);
      }
    });

    test('fixed anchor size produces 1x1 w,h', () {
      const options = SSDAnchorOptions(
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

      final anchors = generateAnchors(options);
      for (final anchor in anchors) {
        expect(anchor[2], 1.0); // w
        expect(anchor[3], 1.0); // h
      }
    });

    test('non-fixed anchor size produces variable w,h', () {
      const options = SSDAnchorOptions(
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
        fixedAnchorSize: false,
      );

      final anchors = generateAnchors(options);
      // With non-fixed, some anchors should have w,h != 1.0
      final hasNonUnit = anchors.any((a) => a[2] != 1.0 || a[3] != 1.0);
      expect(hasNonUnit, true);
    });

    test('anchor centers are within [0, 1] range', () {
      const options = SSDAnchorOptions(
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

      final anchors = generateAnchors(options);
      for (final anchor in anchors) {
        expect(anchor[0], greaterThanOrEqualTo(0.0)); // cx
        expect(anchor[0], lessThanOrEqualTo(1.0));
        expect(anchor[1], greaterThanOrEqualTo(0.0)); // cy
        expect(anchor[1], lessThanOrEqualTo(1.0));
      }
    });

    test('reduceBoxesInLowestLayer produces different count', () {
      const optionsReduced = SSDAnchorOptions(
        numLayers: 4,
        minScale: 0.1484375,
        maxScale: 0.75,
        inputSizeHeight: 192,
        inputSizeWidth: 192,
        anchorOffsetX: 0.5,
        anchorOffsetY: 0.5,
        strides: [8, 16, 16, 16],
        aspectRatios: [1.0],
        reduceBoxesInLowestLayer: true,
        interpolatedScaleAspectRatio: 1.0,
        fixedAnchorSize: true,
      );

      const optionsNormal = SSDAnchorOptions(
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

      final reducedAnchors = generateAnchors(optionsReduced);
      final normalAnchors = generateAnchors(optionsNormal);

      // With reduced boxes in lowest layer, layer 0 gets 3 aspect ratios
      // instead of the normal pattern, so counts differ
      expect(reducedAnchors.length, isNot(normalAnchors.length));
    });

    test('no interpolated scale aspect ratio produces fewer anchors', () {
      const optionsWithInterp = SSDAnchorOptions(
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

      const optionsNoInterp = SSDAnchorOptions(
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
        interpolatedScaleAspectRatio: 0.0, // disabled
        fixedAnchorSize: true,
      );

      final withInterp = generateAnchors(optionsWithInterp);
      final noInterp = generateAnchors(optionsNoInterp);

      // Without interpolation, fewer anchors per position
      expect(noInterp.length, lessThan(withInterp.length));
    });

    test('single stride layer', () {
      const options = SSDAnchorOptions(
        numLayers: 1,
        minScale: 0.2,
        maxScale: 0.9,
        inputSizeHeight: 128,
        inputSizeWidth: 128,
        anchorOffsetX: 0.5,
        anchorOffsetY: 0.5,
        strides: [8],
        aspectRatios: [1.0],
        reduceBoxesInLowestLayer: false,
        interpolatedScaleAspectRatio: 0.0,
        fixedAnchorSize: true,
      );

      final anchors = generateAnchors(options);
      // 128/8 = 16x16 = 256 positions, 1 anchor each
      expect(anchors.length, 256);
    });

    test('multiple aspect ratios', () {
      const options = SSDAnchorOptions(
        numLayers: 1,
        minScale: 0.2,
        maxScale: 0.9,
        inputSizeHeight: 64,
        inputSizeWidth: 64,
        anchorOffsetX: 0.5,
        anchorOffsetY: 0.5,
        strides: [8],
        aspectRatios: [1.0, 2.0, 0.5],
        reduceBoxesInLowestLayer: false,
        interpolatedScaleAspectRatio: 0.0,
        fixedAnchorSize: true,
      );

      final anchors = generateAnchors(options);
      // 64/8 = 8x8 = 64 positions, 3 anchors each
      expect(anchors.length, 64 * 3);
    });
  });

  group('normalizeRadians', () {
    test('zero stays zero', () {
      expect(PalmDetector.normalizeRadians(0.0), closeTo(0.0, 0.0001));
    });

    test('pi stays pi (approximately)', () {
      // pi is the boundary, result should be close to -pi or pi
      final result = PalmDetector.normalizeRadians(math.pi);
      expect(result.abs(), closeTo(math.pi, 0.0001));
    });

    test('values within [-pi, pi] stay unchanged', () {
      expect(PalmDetector.normalizeRadians(1.0), closeTo(1.0, 0.0001));
      expect(PalmDetector.normalizeRadians(-1.0), closeTo(-1.0, 0.0001));
      expect(PalmDetector.normalizeRadians(2.0), closeTo(2.0, 0.0001));
      expect(PalmDetector.normalizeRadians(-2.0), closeTo(-2.0, 0.0001));
    });

    test('values outside range are wrapped', () {
      // 4.0 > pi, should wrap to 4.0 - 2*pi ≈ -2.283
      expect(
          PalmDetector.normalizeRadians(4.0), closeTo(4.0 - 2 * math.pi, 0.01));

      // -4.0 < -pi, should wrap to -4.0 + 2*pi ≈ 2.283
      expect(PalmDetector.normalizeRadians(-4.0),
          closeTo(-4.0 + 2 * math.pi, 0.01));
    });

    test('2*pi wraps to approximately 0', () {
      expect(PalmDetector.normalizeRadians(2 * math.pi), closeTo(0.0, 0.0001));
    });

    test('-2*pi wraps to approximately 0', () {
      expect(PalmDetector.normalizeRadians(-2 * math.pi), closeTo(0.0, 0.0001));
    });

    test('large positive values wrap correctly', () {
      // 10*pi should wrap to 0
      expect(PalmDetector.normalizeRadians(10 * math.pi), closeTo(0.0, 0.001));
    });

    test('large negative values wrap correctly', () {
      expect(PalmDetector.normalizeRadians(-10 * math.pi), closeTo(0.0, 0.001));
    });
  });

  group('PalmDetector constructor', () {
    test('default score threshold is 0.45', () {
      final detector = PalmDetector();
      expect(detector.scoreThreshold, 0.45);
    });

    test('custom score threshold', () {
      final detector = PalmDetector(scoreThreshold: 0.8);
      expect(detector.scoreThreshold, 0.8);
    });

    test('isInitialized is false before initialization', () {
      final detector = PalmDetector();
      expect(detector.isInitialized, false);
    });
  });

  group('PalmDetector lifecycle', () {
    late PalmDetector detector;

    setUp(() {
      detector = PalmDetector();
    });

    test('throws StateError when detectOnMat called before init', () {
      final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);
      try {
        expect(
          () => detector.detectOnMat(mat),
          throwsA(isA<StateError>()),
        );
      } finally {
        mat.dispose();
      }
    });

    test('dispose before init does not throw', () async {
      await detector.dispose();
      expect(detector.isInitialized, false);
    });

    test('initializeFromBuffer works with real model', () async {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_detection.tflite';
      final modelBytes = File(modelPath).readAsBytesSync();

      await detector.initializeFromBuffer(Uint8List.fromList(modelBytes));

      expect(detector.isInitialized, true);
      expect(detector.inputWidth, 192);
      expect(detector.inputHeight, 192);
      expect(detector.anchorsForTest.length, 2016);

      await detector.dispose();
      expect(detector.isInitialized, false);
    });

    test('re-initialization disposes previous state', () async {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_detection.tflite';
      final modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());

      await detector.initializeFromBuffer(modelBytes);
      expect(detector.isInitialized, true);

      // Re-initialize
      await detector.initializeFromBuffer(modelBytes);
      expect(detector.isInitialized, true);

      await detector.dispose();
    });

    test('multiple disposes do not throw', () async {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_detection.tflite';
      final modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());

      await detector.initializeFromBuffer(modelBytes);
      await detector.dispose();
      await detector.dispose();
      expect(detector.isInitialized, false);
    });
  });

  group('PalmDetector detection', () {
    late PalmDetector detector;

    setUpAll(() async {
      detector = PalmDetector();
      final modelPath =
          '${Directory.current.path}/assets/models/hand_detection.tflite';
      final modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());
      await detector.initializeFromBuffer(modelBytes);
    });

    tearDownAll(() async {
      await detector.dispose();
    });

    test('detects palm in hand image', () async {
      final imageBytes = File(
        '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg',
      ).readAsBytesSync();
      final mat = cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);

      try {
        final palms = await detector.detectOnMat(mat);

        expect(palms, isNotEmpty);
        expect(palms.first.score, greaterThan(0.5));
        expect(palms.first.sqnRrSize, greaterThan(0));
        expect(palms.first.sqnRrCenterX, greaterThan(0));
        expect(palms.first.sqnRrCenterY, greaterThan(0));
      } finally {
        mat.dispose();
      }
    });

    test('returns empty for blank image', () async {
      final mat = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);

      try {
        final palms = await detector.detectOnMat(mat);
        expect(palms, isEmpty);
      } finally {
        mat.dispose();
      }
    });

    test('detects multiple palms in two-hands image', () async {
      final imageBytes = File(
        '${Directory.current.path}/assets/samples/2-hands.png',
      ).readAsBytesSync();
      final mat = cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);

      try {
        final palms = await detector.detectOnMat(mat);
        expect(palms.length, greaterThanOrEqualTo(2));
      } finally {
        mat.dispose();
      }
    });

    test('NMS returns palms sorted by score descending', () async {
      final imageBytes = File(
        '${Directory.current.path}/assets/samples/2-hands.png',
      ).readAsBytesSync();
      final mat = cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);

      try {
        final palms = await detector.detectOnMat(mat);

        for (int i = 1; i < palms.length; i++) {
          expect(palms[i].score, lessThanOrEqualTo(palms[i - 1].score));
        }
      } finally {
        mat.dispose();
      }
    });

    test('handles small image', () async {
      final mat = cv.Mat.zeros(10, 10, cv.MatType.CV_8UC3);

      try {
        // Should not crash
        final palms = await detector.detectOnMat(mat);
        expect(palms, isNotNull);
      } finally {
        mat.dispose();
      }
    });

    test('handles portrait image', () async {
      final mat = cv.Mat.zeros(640, 480, cv.MatType.CV_8UC3);

      try {
        final palms = await detector.detectOnMat(mat);
        expect(palms, isNotNull);
      } finally {
        mat.dispose();
      }
    });

    test('handles landscape image', () async {
      final mat = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);

      try {
        final palms = await detector.detectOnMat(mat);
        expect(palms, isNotNull);
      } finally {
        mat.dispose();
      }
    });

    test('palm detections have valid rotation', () async {
      final imageBytes = File(
        '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg',
      ).readAsBytesSync();
      final mat = cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);

      try {
        final palms = await detector.detectOnMat(mat);
        expect(palms, isNotEmpty);

        for (final palm in palms) {
          // Rotation should be in [-pi, pi] after normalizeRadians
          expect(palm.rotation, greaterThanOrEqualTo(-math.pi));
          expect(palm.rotation, lessThanOrEqualTo(math.pi));
        }
      } finally {
        mat.dispose();
      }
    });
  });

  group('PalmDetector portrait vs landscape image handling', () {
    late PalmDetector detector;

    setUpAll(() async {
      detector = PalmDetector();
      final modelPath =
          '${Directory.current.path}/assets/models/hand_detection.tflite';
      final modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());
      await detector.initializeFromBuffer(modelBytes);
    });

    tearDownAll(() async {
      await detector.dispose();
    });

    test('portrait image (height > width) exercises portrait padding branch',
        () async {
      // Load a real hand image and transpose to make it portrait
      final imageBytes = File(
        '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg',
      ).readAsBytesSync();
      final original =
          cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);

      // Crop to portrait orientation (taller than wide)
      final portraitMat = original.region(cv.Rect(100, 0, 300, original.rows));

      try {
        // This should exercise the _imageHeight > _imageWidth branch in _postprocess
        expect(portraitMat.rows, greaterThan(portraitMat.cols));
        final palms = await detector.detectOnMat(portraitMat);
        // Should not crash; may or may not detect depending on crop
        expect(palms, isNotNull);
      } finally {
        portraitMat.dispose();
        original.dispose();
      }
    });

    test('landscape image (width > height) exercises landscape padding branch',
        () async {
      final imageBytes = File(
        '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg',
      ).readAsBytesSync();
      final original =
          cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);

      // Crop to landscape orientation (wider than tall)
      final landscapeMat = original.region(cv.Rect(0, 100, original.cols, 300));

      try {
        expect(landscapeMat.cols, greaterThan(landscapeMat.rows));
        final palms = await detector.detectOnMat(landscapeMat);
        expect(palms, isNotNull);
      } finally {
        landscapeMat.dispose();
        original.dispose();
      }
    });

    test('square image exercises else branch (height == width)', () async {
      final imageBytes = File(
        '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg',
      ).readAsBytesSync();
      final mat = cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);

      try {
        // 612x612 is square
        expect(mat.rows, mat.cols);
        final palms = await detector.detectOnMat(mat);
        expect(palms, isNotEmpty);
      } finally {
        mat.dispose();
      }
    });
  });

  group('PalmDetector with different thresholds', () {
    test('higher threshold produces fewer or equal detections', () async {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_detection.tflite';
      final modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());

      final lenientDetector = PalmDetector(scoreThreshold: 0.3);
      final strictDetector = PalmDetector(scoreThreshold: 0.8);

      await lenientDetector.initializeFromBuffer(modelBytes);
      await strictDetector.initializeFromBuffer(modelBytes);

      final imageBytes = File(
        '${Directory.current.path}/assets/samples/istockphoto-462908027-612x612.jpg',
      ).readAsBytesSync();
      final mat = cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);

      try {
        final lenientResults = await lenientDetector.detectOnMat(mat);
        final strictResults = await strictDetector.detectOnMat(mat);

        expect(
            lenientResults.length, greaterThanOrEqualTo(strictResults.length));
      } finally {
        mat.dispose();
        await lenientDetector.dispose();
        await strictDetector.dispose();
      }
    });
  });
}
