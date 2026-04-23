import 'dart:math' as math;
import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import '../types.dart';
import '../util/image_utils.dart';
import '../models/palm_detector.dart';
import '../models/hand_landmark_model.dart';
import '../models/gesture_recognizer.dart';

/// Helper class to store preprocessing data for each detected palm inside the isolate.
class _HandCropData {
  final PalmDetection palm;
  final cv.Mat croppedHand;
  final double rotation;
  final double centerX;
  final double centerY;
  final double cropSize;

  _HandCropData({
    required this.palm,
    required this.croppedHand,
    required this.rotation,
    required this.centerX,
    required this.centerY,
    required this.cropSize,
  });

  void dispose() => croppedHand.dispose();
}

/// Direct-mode inference core used inside the hand detection background isolate.
///
/// Holds all TFLite models and runs the full hand detection pipeline on the
/// calling thread. Created inside [HandDetector]'s background isolate by
/// [HandDetector._isolateEntry].
///
/// This class is an internal implementation detail of hand_detection and is
/// not part of the public API.
@internal
class HandDetectorCore {
  PalmDetector? _palm;
  HandLandmarkModelRunner? _lm;
  GestureRecognizer? _gestureRecognizer;

  HandMode _mode = HandMode.boxesAndLandmarks;
  int _maxDetections = 10;
  double _minLandmarkScore = 0.5;

  /// Returns true when the core has been initialized with model data.
  bool get isReady => _palm != null;

  /// Initializes all TFLite models from pre-loaded bytes.
  Future<void> initializeFromBuffers({
    required Uint8List palmDetectionBytes,
    required Uint8List handLandmarkBytes,
    Uint8List? gestureEmbedderBytes,
    Uint8List? gestureClassifierBytes,
    required HandMode mode,
    required int maxDetections,
    required double minLandmarkScore,
    required double detectorConf,
    required int interpreterPoolSize,
    required PerformanceConfig performanceConfig,
    required bool enableGestures,
    required double gestureMinConfidence,
  }) async {
    _mode = mode;
    _maxDetections = maxDetections;
    _minLandmarkScore = minLandmarkScore;

    _palm = PalmDetector(scoreThreshold: detectorConf);
    await _palm!.initializeFromBuffer(
      palmDetectionBytes,
      performanceConfig: performanceConfig,
    );

    _lm = HandLandmarkModelRunner(poolSize: interpreterPoolSize);
    await _lm!.initializeFromBuffer(
      handLandmarkBytes,
      performanceConfig: performanceConfig,
    );

    if (enableGestures &&
        gestureEmbedderBytes != null &&
        gestureClassifierBytes != null) {
      _gestureRecognizer =
          GestureRecognizer(minConfidence: gestureMinConfidence);
      await _gestureRecognizer!.initializeFromBuffers(
        embedderBytes: gestureEmbedderBytes,
        classifierBytes: gestureClassifierBytes,
        performanceConfig: performanceConfig,
      );
    }
  }

  /// Runs hand detection directly on the calling thread.
  Future<List<Hand>> detectDirect(cv.Mat image) async {
    if (_palm == null || _lm == null) {
      throw StateError('HandDetectorCore not initialized.');
    }

    final List<PalmDetection> palms = await _palm!.detectOnMat(image);

    final limitedPalms = palms.length > _maxDetections
        ? palms.sublist(0, _maxDetections)
        : palms;

    if (_mode == HandMode.boxes) {
      return _palmsToHands(image, limitedPalms);
    }

    final cropDataList = <_HandCropData>[];
    for (final palm in limitedPalms) {
      final cropped = ImageUtils.rotateAndCropRectangle(image, palm);
      if (cropped == null) continue;

      final (:cx, :cy, :size) =
          ImageUtils.palmCoordinates(palm, image.cols, image.rows);

      cropDataList.add(_HandCropData(
        palm: palm,
        croppedHand: cropped,
        rotation: palm.rotation,
        centerX: cx,
        centerY: cy,
        cropSize: size,
      ));
    }

    final futures = cropDataList.map((data) async {
      try {
        return await _lm!.run(data.croppedHand);
      } catch (_) {
        return null;
      }
    }).toList();

    final allLandmarks = await Future.wait(futures);
    final results = await _buildResults(image, cropDataList, allLandmarks);

    for (final data in cropDataList) {
      data.dispose();
    }

    return results;
  }

  List<Hand> _palmsToHands(cv.Mat image, List<PalmDetection> palms) {
    final results = <Hand>[];
    for (final palm in palms) {
      final (:cx, :cy, :size) =
          ImageUtils.palmCoordinates(palm, image.cols, image.rows);
      final halfSize = size / 2;
      results.add(Hand(
        boundingBox:
            _clampedBoundingBox(cx, cy, halfSize, image.cols, image.rows),
        score: palm.score,
        landmarks: const [],
        imageWidth: image.cols,
        imageHeight: image.rows,
        handedness: null,
        rotation: palm.rotation,
        rotatedCenterX: cx,
        rotatedCenterY: cy,
        rotatedSize: size,
      ));
    }
    return results;
  }

  Future<List<Hand>> _buildResults(
    cv.Mat image,
    List<_HandCropData> cropDataList,
    List<HandLandmarks?> allLandmarks,
  ) async {
    final results = <Hand>[];

    for (int i = 0; i < cropDataList.length; i++) {
      final data = cropDataList[i];
      final lms = allLandmarks[i];

      if (lms == null || lms.score < _minLandmarkScore) continue;

      final transformedLandmarks = <HandLandmark>[];
      final cropW = data.croppedHand.cols.toDouble();
      final cropH = data.croppedHand.rows.toDouble();
      final cosR = math.cos(data.rotation);
      final sinR = math.sin(data.rotation);

      for (final lm in lms.landmarks) {
        final xRel = lm.x - cropW / 2;
        final yRel = lm.y - cropH / 2;
        final xRot = xRel * cosR - yRel * sinR;
        final yRot = xRel * sinR + yRel * cosR;
        final xOrig = xRot + data.centerX;
        final yOrig = yRot + data.centerY;

        transformedLandmarks.add(HandLandmark(
          type: lm.type,
          x: xOrig.clamp(0, image.cols.toDouble()),
          y: yOrig.clamp(0, image.rows.toDouble()),
          z: lm.z,
          visibility: lm.visibility,
        ));
      }

      GestureResult? gesture;
      if (_gestureRecognizer != null && _gestureRecognizer!.isInitialized) {
        gesture = await _gestureRecognizer!.recognize(
          landmarks: transformedLandmarks,
          worldLandmarks: lms.worldLandmarks,
          handedness: lms.handedness,
          imageWidth: image.cols,
          imageHeight: image.rows,
        );
      }

      final halfSize = data.cropSize / 2;
      results.add(Hand(
        boundingBox: _clampedBoundingBox(
            data.centerX, data.centerY, halfSize, image.cols, image.rows),
        score: data.palm.score,
        landmarks: transformedLandmarks,
        imageWidth: image.cols,
        imageHeight: image.rows,
        handedness: lms.handedness,
        rotation: data.rotation,
        rotatedCenterX: data.centerX,
        rotatedCenterY: data.centerY,
        rotatedSize: data.cropSize,
        gesture: gesture,
      ));
    }

    return results;
  }

  BoundingBox _clampedBoundingBox(
      double centerX, double centerY, double halfSize, int imgW, int imgH) {
    return BoundingBox.ltrb(
      (centerX - halfSize).clamp(0, imgW.toDouble()),
      (centerY - halfSize).clamp(0, imgH.toDouble()),
      (centerX + halfSize).clamp(0, imgW.toDouble()),
      (centerY + halfSize).clamp(0, imgH.toDouble()),
    );
  }

  /// Disposes all model resources.
  Future<void> dispose() async {
    await _palm?.dispose();
    await _lm?.dispose();
    if (_gestureRecognizer != null) {
      await _gestureRecognizer!.dispose();
    }
    _palm = null;
    _lm = null;
    _gestureRecognizer = null;
  }
}
