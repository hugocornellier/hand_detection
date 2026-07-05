// Pure-Dart palm-detection geometry shared by the native (OpenCV-backed) and
// web (Canvas-backed) implementations. None of this depends on opencv_dart or
// dart:io, so it compiles on every platform and is the single source of truth
// for SSD anchor generation, box decoding, and the rotation-rectangle
// post-processing that turns raw model output into [PalmDetection]s.

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart'
    show
        SSDAnchorOptions,
        generateAnchors,
        sigmoidClipped,
        normalizeRadians,
        weightedNms;

/// A detected palm with rotation rectangle parameters.
///
/// Used to crop and rotate hand regions for landmark extraction.
class PalmDetection {
  /// Size of the square rotation rectangle (normalized).
  final double sqnRrSize;

  /// Rotation angle in radians.
  final double rotation;

  /// Center X coordinate (normalized 0-1).
  final double sqnRrCenterX;

  /// Center Y coordinate (normalized 0-1).
  final double sqnRrCenterY;

  /// Detection confidence score (0.0 to 1.0).
  final double score;

  /// Creates a palm detection result.
  const PalmDetection({
    required this.sqnRrSize,
    required this.rotation,
    required this.sqnRrCenterX,
    required this.sqnRrCenterY,
    required this.score,
  });
}

/// Generates the palm-detection SSD anchors for the given square input size.
List<List<double>> buildPalmAnchors(int inH, int inW) {
  return generateAnchors(SSDAnchorOptions(
    numLayers: 4,
    minScale: 0.1484375,
    maxScale: 0.75,
    inputSizeHeight: inH,
    inputSizeWidth: inW,
    anchorOffsetX: 0.5,
    anchorOffsetY: 0.5,
    strides: [8, 16, 16, 16],
    aspectRatios: [1.0],
    reduceBoxesInLowestLayer: false,
    interpolatedScaleAspectRatio: 1.0,
    fixedAnchorSize: true,
  ));
}

/// Decodes raw box predictions using anchors.
///
/// Returns decoded boxes as `[score, cx, cy, boxSize, kp0X, kp0Y, kp2X, kp2Y]`.
/// Reads directly from flat [Float32List] outputs to avoid the boxed-double
/// overhead of nested lists. Raw box layout per anchor is
/// `[cx, cy, w, h, kp0_x, kp0_y, ...]` where each value is offset relative to
/// the anchor and scaled by [scale].
List<List<double>> decodePalmBoxes(
  Float32List rawBoxes,
  Float32List rawScores,
  List<List<double>> anchors,
  int boxStride,
  double scoreThreshold, {
  double scale = 192.0,
}) {
  final results = <List<double>>[];
  final invScale = 1.0 / scale;
  final numAnchors = rawScores.length;
  final stride = boxStride;

  for (int i = 0; i < numAnchors; i++) {
    final rawScore = rawScores[i];
    final score = sigmoidClipped(rawScore);

    if (score <= scoreThreshold) continue;

    final base = i * stride;
    final anchor = anchors[i];

    final anchorW = anchor[2];
    final anchorH = anchor[3];
    final anchorX = anchor[0];
    final anchorY = anchor[1];

    final cx = rawBoxes[base] * anchorW * invScale + anchorX;
    final cy = rawBoxes[base + 1] * anchorH * invScale + anchorY;

    final wPoint = rawBoxes[base + 2] * anchorW * invScale + anchorX;
    final hPoint = rawBoxes[base + 3] * anchorH * invScale + anchorY;
    final w = wPoint - anchorX;
    final h = hPoint - anchorY;
    final boxSize = math.max(w, h);

    final kp0X = rawBoxes[base + 4] * anchorW * invScale + anchorX;
    final kp0Y = rawBoxes[base + 5] * anchorH * invScale + anchorY;

    final kp2X = rawBoxes[base + 8] * anchorW * invScale + anchorX;
    final kp2Y = rawBoxes[base + 9] * anchorH * invScale + anchorY;

    results.add([score, cx, cy, boxSize, kp0X, kp0Y, kp2X, kp2Y]);
  }

  return results;
}

/// Post-processes decoded boxes into palm detections.
///
/// Transforms coordinates from model space back to original image space,
/// accounting for the padding applied during preprocessing, then runs weighted
/// non-maximum suppression. Matches the Python reference implementation.
List<PalmDetection> postprocessPalms(
  List<List<double>> boxes, {
  required int imageWidth,
  required int imageHeight,
  required int squareStandardSize,
  required int squarePaddingHalfSize,
  double roiScale = 2.6,
  double iouThreshold = 0.45,
}) {
  if (boxes.isEmpty) return [];

  final palms = <PalmDetection>[];

  for (final box in boxes) {
    final pdScore = box[0];
    final boxX = box[1];
    final boxY = box[2];
    final boxSize = box[3];
    final kp0X = box[4];
    final kp0Y = box[5];
    final kp2X = box[6];
    final kp2Y = box[7];

    if (boxSize > 0) {
      final kp02X = kp2X - kp0X;
      final kp02Y = kp2Y - kp0Y;
      // MediaPipe palm_detection_detection_to_roi RectTransformationCalculator:
      // scale_x/scale_y ([roiScale], default 2.6) on the square_long palm box,
      // with shift_y -0.5 (the center shift below). The Python reference port
      // this file was adapted from used 2.9; 2.6 is the value MediaPipe's
      // shipped hand_landmark_full model was trained against.
      final sqnRrSize = roiScale * boxSize;
      var rotation = 0.5 * math.pi - math.atan2(-kp02Y, kp02X);
      rotation = normalizeRadians(rotation);
      var sqnRrCenterX = boxX + 0.5 * boxSize * math.sin(rotation);
      var sqnRrCenterY = boxY - 0.5 * boxSize * math.cos(rotation);

      if (imageHeight > imageWidth) {
        sqnRrCenterX =
            (sqnRrCenterX * squareStandardSize - squarePaddingHalfSize) /
                imageWidth;
      } else {
        sqnRrCenterY =
            (sqnRrCenterY * squareStandardSize - squarePaddingHalfSize) /
                imageHeight;
      }

      palms.add(PalmDetection(
        sqnRrSize: sqnRrSize,
        rotation: rotation,
        sqnRrCenterX: sqnRrCenterX,
        sqnRrCenterY: sqnRrCenterY,
        score: pdScore,
      ));
    }
  }

  return _nmsPalms(palms, imageWidth, imageHeight, iouThreshold);
}

/// Landmark subset used by MediaPipe's HandLandmarksToRectCalculator: wrist,
/// the three thumb base joints, and the MCP + PIP joints of the four fingers.
/// Fingertips and DIP joints are deliberately excluded so the ROI follows the
/// stable knuckle region rather than fast-moving fingertips.
const List<int> _roiLandmarkIndices = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];

/// Builds the next-frame tracking ROI for a hand from its 21 landmarks,
/// porting MediaPipe's landmarks-to-ROI step of the hand tracking graph
/// (HandLandmarksToRectCalculator followed by RectTransformationCalculator
/// with scale 2.0, shift_y -0.1, square_long):
///
/// 1. Orient by the axis from the wrist to the weighted mean of the index,
///    middle and ring MCP knuckles (middle counted twice), so that axis points
///    "up".
/// 2. Take the tight bounding box of the knuckle-region landmark subset
///    (fingertips and DIPs excluded) in that rotated frame.
/// 3. Shift the box by [shiftY] of its own unscaled height along the hand
///    axis, make it square on its long side, and expand it by [scale] for
///    inter-frame motion margin.
/// 4. Re-normalise into [PalmDetection] space so the standard crop path can
///    consume it on the next frame.
///
/// [xs]/[ys] are the landmark pixel coordinates in original image space, in
/// the standard 21-landmark order. [shiftY] follows MediaPipe's sign
/// convention: negative shifts toward the fingertips. Returns null when the
/// landmarks cannot produce a usable ROI: fewer than 21 points, a degenerate
/// box, or a size outside [minSqnSize]..[maxSqnSize] (normalised by the
/// longest image side), so the caller drops tracking and the palm detector
/// re-acquires instead of perpetuating a bad region.
PalmDetection? roiFromHandLandmarks({
  required List<double> xs,
  required List<double> ys,
  required int imageWidth,
  required int imageHeight,
  required double score,
  double scale = 2.0,
  double shiftY = -0.1,
  double minSqnSize = 0.03,
  double maxSqnSize = 1.2,
}) {
  if (xs.length < 21 || ys.length < 21) return null;

  // Rotation axis: wrist -> ((indexMCP + ringMCP) / 2 + middleMCP) / 2,
  // oriented so it points "up" (MediaPipe's kTargetAngle of pi/2).
  final x1 = ((xs[5] + xs[13]) / 2 + xs[9]) / 2;
  final y1 = ((ys[5] + ys[13]) / 2 + ys[9]) / 2;
  final rotation =
      normalizeRadians(0.5 * math.pi - math.atan2(-(y1 - ys[0]), x1 - xs[0]));
  final cosR = math.cos(rotation);
  final sinR = math.sin(rotation);

  // Axis-aligned bounds of the knuckle-subset landmarks.
  double aaMinX = double.infinity, aaMinY = double.infinity;
  double aaMaxX = -double.infinity, aaMaxY = -double.infinity;
  for (final i in _roiLandmarkIndices) {
    if (xs[i] < aaMinX) aaMinX = xs[i];
    if (xs[i] > aaMaxX) aaMaxX = xs[i];
    if (ys[i] < aaMinY) aaMinY = ys[i];
    if (ys[i] > aaMaxY) aaMaxY = ys[i];
  }
  final aaCx = (aaMinX + aaMaxX) / 2;
  final aaCy = (aaMinY + aaMaxY) / 2;

  // Bounds of the same landmarks in the hand-aligned frame: rotate each
  // delta from the axis-aligned center by R(-rotation).
  double minX = double.infinity, minY = double.infinity;
  double maxX = -double.infinity, maxY = -double.infinity;
  for (final i in _roiLandmarkIndices) {
    final dx = xs[i] - aaCx;
    final dy = ys[i] - aaCy;
    final rx = dx * cosR + dy * sinR;
    final ry = -dx * sinR + dy * cosR;
    if (rx < minX) minX = rx;
    if (rx > maxX) maxX = rx;
    if (ry < minY) minY = ry;
    if (ry > maxY) maxY = ry;
  }
  final width = maxX - minX;
  final height = maxY - minY;

  // Map the rotated-frame box center back to image space:
  // R(+rotation) * projectedCenter + axisAlignedCenter.
  final projCx = (minX + maxX) / 2;
  final projCy = (minY + maxY) / 2;
  var cx = projCx * cosR - projCy * sinR + aaCx;
  var cy = projCx * sinR + projCy * cosR + aaCy;

  // RectTransformationCalculator: shift by the unscaled height along the
  // rect's own axes (shift_x is 0), then square the long side and scale.
  cx += -height * shiftY * sinR;
  cy += height * shiftY * cosR;
  final size = math.max(width, height) * scale;
  if (size <= 0) return null;

  final maxDim = math.max(imageWidth, imageHeight);
  final sqnSize = size / maxDim;
  if (sqnSize < minSqnSize || sqnSize > maxSqnSize) return null;

  return PalmDetection(
    sqnRrSize: sqnSize,
    rotation: rotation,
    sqnRrCenterX: cx / imageWidth,
    sqnRrCenterY: cy / imageHeight,
    score: score,
  );
}

/// Axis-aligned IoU of two ROIs in pixel space, ignoring rotation, mirroring
/// MediaPipe's AssociationNormRectCalculator (whose rect conversion also drops
/// rotation). [PalmDetection] ROIs are squares in pixel space, and IoU of
/// axis-aligned boxes is invariant under the per-axis normalisation MediaPipe
/// applies, so this equals MediaPipe's normalised-space overlap similarity.
double roiIou(
    PalmDetection a, PalmDetection b, int imageWidth, int imageHeight) {
  final maxDim = math.max(imageWidth, imageHeight).toDouble();
  final aHalf = a.sqnRrSize * maxDim / 2;
  final bHalf = b.sqnRrSize * maxDim / 2;
  final aCx = a.sqnRrCenterX * imageWidth;
  final aCy = a.sqnRrCenterY * imageHeight;
  final bCx = b.sqnRrCenterX * imageWidth;
  final bCy = b.sqnRrCenterY * imageHeight;
  final iw =
      math.min(aCx + aHalf, bCx + bHalf) - math.max(aCx - aHalf, bCx - bHalf);
  final ih =
      math.min(aCy + aHalf, bCy + bHalf) - math.max(aCy - aHalf, bCy - bHalf);
  if (iw <= 0 || ih <= 0) return 0;
  final inter = iw * ih;
  final union = 4 * aHalf * aHalf + 4 * bHalf * bHalf - inter;
  return union <= 0 ? 0 : inter / union;
}

/// Merges fresh palm-detection ROIs with ROIs tracked from the previous
/// frame's landmarks, porting MediaPipe's AssociationNormRectCalculator
/// (hand tracking graph, min_similarity_threshold 0.5).
///
/// Elements are added in order, [lowPriority] (fresh palm detections) first
/// and [highPriority] (tracked ROIs) after; each new element removes every
/// already-kept element whose [roiIou] exceeds [minSimilarityThreshold].
/// Later entries therefore win, so an overlapping palm/tracked pair keeps the
/// tracked ROI, and entries within the same list are also deduplicated
/// against each other (tracked-vs-tracked included).
List<PalmDetection> associateRois(
  List<PalmDetection> lowPriority,
  List<PalmDetection> highPriority, {
  required int imageWidth,
  required int imageHeight,
  double minSimilarityThreshold = 0.5,
}) {
  final kept = <PalmDetection>[];
  void add(PalmDetection roi) {
    kept.removeWhere((o) =>
        roiIou(o, roi, imageWidth, imageHeight) > minSimilarityThreshold);
    kept.add(roi);
  }

  for (final roi in lowPriority) {
    add(roi);
  }
  for (final roi in highPriority) {
    add(roi);
  }
  return kept;
}

/// Weighted Non-Maximum Suppression for palm detections.
///
/// Fuses overlapping boxes by score-weighted coordinate averaging, producing
/// tighter bounding boxes from the many overlapping SSD anchors that fire on
/// the same palm. Keeps the highest-scoring detection's rotation (derived from
/// keypoints) while averaging center and size.
List<PalmDetection> _nmsPalms(
  List<PalmDetection> palms,
  int imageWidth,
  int imageHeight,
  double iouThreshold,
) {
  if (palms.isEmpty) return palms;
  final sorted = List<PalmDetection>.from(palms)
    ..sort((a, b) => b.score.compareTo(a.score));
  final maxDim = math.max(imageWidth, imageHeight).toDouble();

  List<double> toXYXY(PalmDetection p) {
    final halfSize = (p.sqnRrSize * maxDim) / 2;
    final centerX = p.sqnRrCenterX * imageWidth;
    final centerY = p.sqnRrCenterY * imageHeight;
    return [
      centerX - halfSize,
      centerY - halfSize,
      centerX + halfSize,
      centerY + halfSize,
    ];
  }

  final boxes = sorted.map(toXYXY).toList();
  final scores = sorted.map((p) => p.score).toList();
  final results = weightedNms(boxes, scores, iouThres: iouThreshold);
  return [
    for (final r in results)
      PalmDetection(
        sqnRrSize: math.max(r.box[2] - r.box[0], r.box[3] - r.box[1]) / maxDim,
        rotation: sorted[r.index].rotation,
        sqnRrCenterX: (r.box[0] + r.box[2]) / 2 / imageWidth,
        sqnRrCenterY: (r.box[1] + r.box[3]) / 2 / imageHeight,
        score: r.score,
      ),
  ];
}
