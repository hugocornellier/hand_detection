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
      final sqnRrSize = 2.9 * boxSize;
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
