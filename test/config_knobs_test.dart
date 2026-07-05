// Tests for the tunable model knobs exposed on HandDetector: palmNmsIou,
// palmRoiScale, and TrackingConfig. These guard the wiring end-to-end so a
// future refactor cannot silently drop a parameter somewhere between the public
// API and the model:
//
//   Group 1 (pure Dart): postprocessPalms honors roiScale / iouThreshold.
//   Group 2 (real model): PalmDetector forwards its fields into postprocessPalms.
//   Group 3 (full isolate): palmRoiScale survives the 6-layer + isolate plumbing.
//   Group 4: TrackingConfig defaults are the MediaPipe values, and the config
//            is accepted through the whole tracking path.
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/src/shared/hand_geometry.dart';
import 'package:hand_detection/src/shared/hand_types.dart';
import 'package:hand_detection/src/models/palm_detector.dart';
import 'package:hand_detection/src/hand_detector.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final String root = Directory.current.path;
  Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());

  // ---------------------------------------------------------------------------
  // Group 1: postprocessPalms geometry (pure Dart, no model). Deterministic
  // proof that the two palm knobs mathematically change the output.
  // ---------------------------------------------------------------------------
  group('postprocessPalms honors palm tuning', () {
    // Square image => squarePaddingHalfSize 0 and no center renormalization,
    // so the math stays easy to reason about.
    List<PalmDetection> square(
      List<List<double>> boxes, {
      double roiScale = 2.6,
      double iouThreshold = 0.45,
    }) =>
        postprocessPalms(
          boxes,
          imageWidth: 100,
          imageHeight: 100,
          squareStandardSize: 100,
          squarePaddingHalfSize: 0,
          roiScale: roiScale,
          iouThreshold: iouThreshold,
        );

    test('roiScale scales the ROI size linearly', () {
      // Decoded box layout: [score, cx, cy, boxSize, kp0X, kp0Y, kp2X, kp2Y].
      // Keypoints share an x (kp02X == 0) with kp2 above kp0, so rotation is 0.
      final boxes = [
        [0.9, 0.5, 0.5, 0.2, 0.5, 0.6, 0.5, 0.4],
      ];
      final base = square(boxes, roiScale: 2.6);
      final wide = square(boxes, roiScale: 5.2);

      expect(base, hasLength(1));
      expect(wide, hasLength(1));
      // sqnRrSize == roiScale * boxSize.
      expect(base.first.sqnRrSize, closeTo(2.6 * 0.2, 1e-6));
      expect(wide.first.sqnRrSize, closeTo(5.2 * 0.2, 1e-6));
      expect(wide.first.sqnRrSize / base.first.sqnRrSize, closeTo(2.0, 1e-6));
    });

    test('iouThreshold controls how hard overlapping palms merge', () {
      // Two rotation-0 squares whose axis-aligned IoU is ~0.5. A low threshold
      // suppresses the weaker one (weighted NMS fuses the cluster -> 1 palm); a
      // high threshold leaves both (-> 2 palms).
      List<double> box(double cx, double score) =>
          [score, cx, 0.5, 0.1, cx, 0.55, cx, 0.45];
      final boxes = [box(0.45, 0.9), box(0.5367, 0.85)];

      expect(square(boxes, iouThreshold: 0.1), hasLength(1));
      expect(square(boxes, iouThreshold: 0.9), hasLength(2));
    });
  });

  // ---------------------------------------------------------------------------
  // Group 2: PalmDetector forwards roiScale / nmsIouThreshold into
  // postprocessPalms, exercised against the real palm model.
  // ---------------------------------------------------------------------------
  group('PalmDetector honors roiScale / nmsIouThreshold', () {
    final palmModel = load('$root/assets/models/hand_detection.tflite');
    final imageBytes = load('$root/example/assets/samples/2-hands.png');

    Future<List<PalmDetection>> detect({
      double roiScale = 2.6,
      double nmsIou = 0.45,
    }) async {
      final det = PalmDetector(roiScale: roiScale, nmsIouThreshold: nmsIou);
      await det.initializeFromBuffer(palmModel);
      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      try {
        return await det.detectOnMat(mat);
      } finally {
        mat.dispose();
        await det.dispose();
      }
    }

    test('roiScale scales detected palm ROI size', () async {
      final small = await detect(roiScale: 2.0);
      final large = await detect(roiScale: 4.0);

      expect(small, isNotEmpty);
      expect(large, isNotEmpty);
      expect(large.first.sqnRrSize, greaterThan(small.first.sqnRrSize));
      // Uniform scale -> the top palm grows by roughly the scale ratio (2x).
      expect(large.first.sqnRrSize / small.first.sqnRrSize, greaterThan(1.5));
    });

    test('higher nmsIouThreshold keeps at least as many palms', () async {
      final aggressive = await detect(nmsIou: 0.1);
      final lenient = await detect(nmsIou: 0.95);

      // Weighted NMS is monotonic in the threshold: a more lenient IoU merges
      // less, so it never returns fewer palms than an aggressive one. (An
      // aggressive 0.1 can fuse the two hands' expanded ROIs down to a single
      // palm.) Group 1 covers the exact merge-count change deterministically.
      expect(lenient.length, greaterThanOrEqualTo(aggressive.length));
      // A lenient threshold keeps the near-duplicate per-hand anchors, so it
      // resolves at least the two hands, proving the field reaches the model.
      expect(lenient.length, greaterThanOrEqualTo(2));
    });
  });

  // ---------------------------------------------------------------------------
  // Group 3: palmRoiScale survives the full public -> isolate -> core -> model
  // path. This is the guard against an isolate-plumbing regression that
  // defaults-only tests cannot catch.
  // ---------------------------------------------------------------------------
  group('HandDetector threads palmRoiScale through the isolate', () {
    final palmModel = load('$root/assets/models/hand_detection.tflite');
    final lmModel = load('$root/assets/models/hand_landmark_full.tflite');
    final imageBytes = load('$root/example/assets/samples/2-hands.png');

    Future<List<Hand>> detect(double palmRoiScale) async {
      final det = HandDetector();
      await det.initializeFromBuffers(
        palmDetectionBytes: palmModel,
        handLandmarkBytes: lmModel,
        // Box-only mode keeps the box size proportional to the palm ROI and
        // skips landmark inference, so the effect of palmRoiScale is visible
        // directly in the returned geometry.
        mode: HandMode.boxes,
        palmRoiScale: palmRoiScale,
      );
      try {
        return await det.detect(imageBytes);
      } finally {
        await det.dispose();
      }
    }

    test('larger palmRoiScale yields larger boxes', () async {
      final small = await detect(2.0);
      final large = await detect(4.0);

      expect(small, isNotEmpty);
      expect(large, isNotEmpty);
      expect(small.first.rotatedSize, isNotNull);
      expect(large.first.rotatedSize, isNotNull);
      expect(large.first.rotatedSize!, greaterThan(small.first.rotatedSize!));
    });
  });

  // ---------------------------------------------------------------------------
  // Group 4: TrackingConfig.
  // ---------------------------------------------------------------------------
  group('TrackingConfig', () {
    test('defaults match the MediaPipe hand tracking graph', () {
      const t = TrackingConfig();
      expect(t.roiScale, 2.0);
      expect(t.roiShiftY, -0.1);
      expect(t.associationIou, 0.5);
      expect(t.minRoiSize, 0.03);
      expect(t.maxRoiSize, 1.2);
    });

    test('is accepted end-to-end with tracking enabled', () async {
      final det = HandDetector();
      await det.initializeFromBuffers(
        palmDetectionBytes: load('$root/assets/models/hand_detection.tflite'),
        handLandmarkBytes:
            load('$root/assets/models/hand_landmark_full.tflite'),
        mode: HandMode.boxesAndLandmarks,
        enableTracking: true,
        trackingConfig:
            const TrackingConfig(roiScale: 2.4, associationIou: 0.4),
      );
      final img = load('$root/example/assets/samples/2-hands.png');
      try {
        final frame1 = await det.detect(img);
        // Second frame runs the tracking path seeded by frame 1's landmark ROIs.
        final frame2 = await det.detect(img);
        expect(frame1, isNotEmpty);
        expect(frame2, isNotEmpty);
      } finally {
        await det.dispose();
      }
    });
  });
}
