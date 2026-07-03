import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/src/shared/hand_geometry.dart';

/// Builds 21 landmark coordinates with controlled extremes.
///
/// All points default to ([fillX], [fillY]); callers override individual
/// indices to shape the box and the orientation axis. MediaPipe's
/// HandLandmarksToRectCalculator computes the rect over only the knuckle subset
/// {0,1,2,3,5,6,9,10,13,14,17,18}; the rotation axis runs from the wrist (0) to
/// the mean of the index (5), middle (9) and ring (13) MCP joints (middle
/// counted twice). Fingertip/DIP indices (4,7,8,11,12,15,16,19,20) are excluded
/// from the rect, so tests override some of them with wild values to prove they
/// are ignored.
(List<double>, List<double>) buildLandmarks({
  required double fillX,
  required double fillY,
  Map<int, double> xOverrides = const {},
  Map<int, double> yOverrides = const {},
}) {
  final xs = List<double>.filled(21, fillX);
  final ys = List<double>.filled(21, fillY);
  xOverrides.forEach((i, v) => xs[i] = v);
  yOverrides.forEach((i, v) => ys[i] = v);
  return (xs, ys);
}

PalmDetection _palm(
  double cx,
  double cy,
  double size, {
  double score = 1.0,
  double rotation = 0.0,
}) =>
    PalmDetection(
      sqnRrSize: size,
      rotation: rotation,
      sqnRrCenterX: cx,
      sqnRrCenterY: cy,
      score: score,
    );

void main() {
  group('roiFromHandLandmarks', () {
    test('upright hand produces zero rotation and a 2x expanded square', () {
      // Wrist (0) below the MCP joints (5/9/13): the wrist->MCP axis already
      // points "up", so rotation must be ~0 and the local frame == image frame.
      // Knuckle-subset box: x in [80, 120] (w 40, from idx 1 and 17), y in
      // [140, 200] (h 60, from the MCPs at 140 and the wrist at 200). The wild
      // fingertip/DIP overrides (4, 8, 20) must NOT affect the box.
      final (xs, ys) = buildLandmarks(
        fillX: 100,
        fillY: 170,
        xOverrides: {1: 80, 17: 120, 4: 9999, 8: -9999, 20: 5000},
        yOverrides: {
          0: 200,
          5: 140,
          9: 140,
          13: 140,
          4: -9999,
          8: 9999,
          20: 7777
        },
      );

      final roi = roiFromHandLandmarks(
        xs: xs,
        ys: ys,
        imageWidth: 640,
        imageHeight: 480,
        score: 0.9,
        shiftY: 0.0,
      );

      expect(roi, isNotNull);
      expect(roi!.rotation, closeTo(0.0, 1e-9));
      // size = max(40, 60) * 2.0 = 120, normalised by maxDim 640.
      expect(roi.sqnRrSize, closeTo(120 / 640, 1e-9));
      expect(roi.sqnRrCenterX, closeTo(100 / 640, 1e-9));
      expect(roi.sqnRrCenterY, closeTo(170 / 480, 1e-9));
      expect(roi.score, 0.9);
    });

    test('hand pointing right rotates the ROI by 90 degrees', () {
      // Wrist (0) at x=100, MCPs (5/9/13) at x=160, same y: fingers point +x,
      // so the rect rotates by pi/2. Knuckle box: x in [100, 160], y in
      // [90, 110] (from idx 1 and 17). Wild fingertip overrides are ignored.
      final (xs, ys) = buildLandmarks(
        fillX: 130,
        fillY: 100,
        xOverrides: {
          0: 100,
          5: 160,
          9: 160,
          13: 160,
          4: 9999,
          8: -9999,
          20: 5000
        },
        yOverrides: {1: 90, 17: 110, 4: -9999, 8: 9999, 20: 7777},
      );

      final roi = roiFromHandLandmarks(
        xs: xs,
        ys: ys,
        imageWidth: 640,
        imageHeight: 480,
        score: 0.8,
        shiftY: 0.0,
      );

      expect(roi, isNotNull);
      expect(roi!.rotation, closeTo(math.pi / 2, 1e-9));
      // Rotated box: width 20 (y-extent), height 60 (x-extent).
      // size = max(20, 60) * 2.0 = 120.
      expect(roi.sqnRrSize, closeTo(120 / 640, 1e-9));
      expect(roi.sqnRrCenterX, closeTo(130 / 640, 1e-9));
      expect(roi.sqnRrCenterY, closeTo(100 / 480, 1e-9));
    });

    test('default shiftY (-0.1) nudges the ROI toward the fingertips', () {
      // Same upright hand as the first test (box height 60).
      List<double> mkXs() => buildLandmarks(
            fillX: 100,
            fillY: 170,
            xOverrides: {1: 80, 17: 120},
            yOverrides: {0: 200, 5: 140, 9: 140, 13: 140},
          ).$1;
      List<double> mkYs() => buildLandmarks(
            fillX: 100,
            fillY: 170,
            xOverrides: {1: 80, 17: 120},
            yOverrides: {0: 200, 5: 140, 9: 140, 13: 140},
          ).$2;

      final baseline = roiFromHandLandmarks(
        xs: mkXs(),
        ys: mkYs(),
        imageWidth: 640,
        imageHeight: 480,
        score: 0.9,
        shiftY: 0.0,
      );
      final shifted = roiFromHandLandmarks(
        xs: mkXs(),
        ys: mkYs(),
        imageWidth: 640,
        imageHeight: 480,
        score: 0.9,
        // shiftY omitted: uses MediaPipe's -0.1 default.
      );

      // Upright hand: fingertips are "up" (smaller y). The default shift moves
      // the center up by |shiftY| * height = 0.1 * 60 = 6 px.
      expect(baseline!.sqnRrCenterY, closeTo(170 / 480, 1e-9));
      expect(shifted!.sqnRrCenterY, closeTo((170 - 6) / 480, 1e-9));
      expect(shifted.sqnRrCenterY, lessThan(baseline.sqnRrCenterY));
      expect(shifted.sqnRrCenterX, closeTo(baseline.sqnRrCenterX, 1e-9));
    });

    test('returns null for fewer than 21 landmarks', () {
      expect(
        roiFromHandLandmarks(
          xs: List.filled(20, 100),
          ys: List.filled(20, 100),
          imageWidth: 640,
          imageHeight: 480,
          score: 0.9,
        ),
        isNull,
      );
    });

    test('returns null for a degenerate (zero-size) landmark set', () {
      final (xs, ys) = buildLandmarks(fillX: 50, fillY: 50);
      expect(
        roiFromHandLandmarks(
          xs: xs,
          ys: ys,
          imageWidth: 640,
          imageHeight: 480,
          score: 0.9,
        ),
        isNull,
      );
    });

    test('returns null when the ROI is too small to track', () {
      // 5px landmark spread -> 10px ROI = 10/640, below the 0.03 floor.
      final (xs, ys) = buildLandmarks(
        fillX: 100,
        fillY: 100,
        yOverrides: {0: 103, 9: 98},
      );
      expect(
        roiFromHandLandmarks(
          xs: xs,
          ys: ys,
          imageWidth: 640,
          imageHeight: 480,
          score: 0.9,
        ),
        isNull,
      );
    });

    test('returns null for a runaway ROI larger than the frame', () {
      // 2000px spread -> 4000px ROI in a 640px frame, above the 1.2 ceiling.
      final (xs, ys) = buildLandmarks(
        fillX: 100,
        fillY: 1000,
        yOverrides: {0: 2000, 9: 0},
      );
      expect(
        roiFromHandLandmarks(
          xs: xs,
          ys: ys,
          imageWidth: 640,
          imageHeight: 480,
          score: 0.9,
        ),
        isNull,
      );
    });
  });

  group('roiIou', () {
    test('identical ROIs have IoU 1.0', () {
      final a = _palm(0.5, 0.5, 0.4);
      expect(roiIou(a, a, 100, 100), closeTo(1.0, 1e-9));
    });

    test('half-overlapping ROIs give the exact area ratio', () {
      // 100x100 image, size 0.4 -> 40px squares. A spans [30,70], B [50,90] in
      // x, both [30,70] in y. inter 20*40=800, union 1600+1600-800=2400.
      final a = _palm(0.5, 0.5, 0.4);
      final b = _palm(0.7, 0.5, 0.4);
      expect(roiIou(a, b, 100, 100), closeTo(800 / 2400, 1e-9));
    });

    test('disjoint ROIs have IoU 0', () {
      final a = _palm(0.5, 0.5, 0.4); // [30,70]
      final c = _palm(0.95, 0.5, 0.4); // [75,115] -> no x overlap
      expect(roiIou(a, c, 100, 100), 0);
    });

    test('IoU is computed in pixel space on a non-square image', () {
      // 200x100 image (maxDim 200), size 0.2 -> 40px squares. Centers 0.5/0.6
      // in x map to 100px/120px; the boxes are [80,120] and [100,140] in x,
      // [30,70] in y. Same 1/3 ratio as the square case, which the old
      // normalised-space IoU would have gotten wrong.
      final a = _palm(0.5, 0.5, 0.2);
      final b = _palm(0.6, 0.5, 0.2);
      expect(roiIou(a, b, 200, 100), closeTo(800 / 2400, 1e-9));
    });
  });

  group('associateRois', () {
    test('a tracked ROI wins over an overlapping fresh palm', () {
      // IoU(palm, tracked) ~0.9 > 0.5: the palm is dropped, tracked kept.
      final palm = _palm(0.5, 0.5, 0.4, score: 0.9);
      final tracked = _palm(0.52, 0.5, 0.4, score: 0.5);
      final out =
          associateRois([palm], [tracked], imageWidth: 100, imageHeight: 100);
      expect(out.length, 1);
      expect(out.single.score, 0.5); // the tracked ROI
      expect(out.single.sqnRrCenterX, 0.52);
    });

    test('non-overlapping palm and tracked ROIs are both kept', () {
      final palm = _palm(0.2, 0.2, 0.2);
      final tracked = _palm(0.8, 0.8, 0.2);
      final out =
          associateRois([palm], [tracked], imageWidth: 100, imageHeight: 100);
      expect(out.length, 2);
    });

    test('deduplicates overlapping tracked ROIs against each other', () {
      // The behaviour MediaPipe has and the old one-directional check lacked:
      // two tracked ROIs that drift together collapse to the later one.
      final t1 = _palm(0.5, 0.5, 0.4, score: 1);
      final t2 = _palm(0.52, 0.5, 0.4, score: 2);
      final out =
          associateRois(const [], [t1, t2], imageWidth: 100, imageHeight: 100);
      expect(out.length, 1);
      expect(out.single.score, 2); // later element wins
    });

    test('ROIs overlapping below the threshold are all kept', () {
      // IoU 1/3 < 0.5.
      final a = _palm(0.5, 0.5, 0.4);
      final b = _palm(0.7, 0.5, 0.4);
      final out = associateRois([a], [b], imageWidth: 100, imageHeight: 100);
      expect(out.length, 2);
    });
  });
}
