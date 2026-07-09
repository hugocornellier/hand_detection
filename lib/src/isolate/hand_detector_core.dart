import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_litert/flutter_litert.dart'
    show Accelerator, BoundingBox, PerformanceConfig, Precision;
import 'package:meta/meta.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import '../types.dart';
import '../util/image_utils.dart';
import '../shared/hand_geometry.dart' show associateRois, roiFromHandLandmarks;
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
  bool _enableTracking = false;

  // MediaPipe-style detection + tracking. When enabled, each kept hand's
  // landmark-derived rotated ROI is carried to the next frame and landmarked
  // directly, so a hand persists without the palm detector re-finding it on
  // every frame (re-detecting a small, deformable palm is what makes the
  // overlay blink on hard frames). The palm detector then only runs to discover
  // new hands, or whenever we are tracking fewer than [_maxDetections].
  List<PalmDetection> _trackedRois = const [];
  List<PalmDetection> _nextTrackedRois = const [];

  // Landmark-derived ROI tuning (MediaPipe hand_landmark_landmarks_to_roi.pbtxt
  // RectTransformationCalculator + AssociationNormRectCalculator), configurable
  // via [TrackingConfig]. The defaults expand the tracked square and nudge it
  // toward the fingertips exactly as MediaPipe does, and treat a fresh
  // palm-derived ROI overlapping a tracked one above [associationIou] as the
  // same hand. Set in [initializeFromBuffers]; only consulted when tracking is
  // enabled.
  TrackingConfig _tracking = const TrackingConfig();

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
    double palmNmsIou = 0.3,
    double palmRoiScale = 2.6,
    bool enableTracking = false,
    TrackingConfig trackingConfig = const TrackingConfig(),
    required int interpreterPoolSize,
    required PerformanceConfig performanceConfig,
    required bool enableGestures,
    required double gestureMinConfidence,
    bool useCompiledModel = false,
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    _mode = mode;
    _maxDetections = maxDetections;
    _minLandmarkScore = minLandmarkScore;
    _enableTracking = enableTracking;
    _tracking = trackingConfig;
    resetTracking();

    _palm = PalmDetector(
      scoreThreshold: detectorConf,
      nmsIouThreshold: palmNmsIou,
      roiScale: palmRoiScale,
    );
    if (useCompiledModel) {
      await _palm!.initializeCompiledFromBuffer(
        palmDetectionBytes,
        accelerators: accelerators,
        precision: precision,
      );
    } else {
      await _palm!.initializeFromBuffer(
        palmDetectionBytes,
        performanceConfig: performanceConfig,
      );
    }

    _lm = HandLandmarkModelRunner(poolSize: interpreterPoolSize);
    if (useCompiledModel) {
      await _lm!.initializeCompiledFromBuffer(
        handLandmarkBytes,
        accelerators: accelerators,
        precision: precision,
      );
    } else {
      await _lm!.initializeFromBuffer(
        handLandmarkBytes,
        performanceConfig: performanceConfig,
      );
    }

    if (enableGestures &&
        gestureEmbedderBytes != null &&
        gestureClassifierBytes != null) {
      _gestureRecognizer =
          GestureRecognizer(minConfidence: gestureMinConfidence);
      if (useCompiledModel) {
        await _gestureRecognizer!.initializeCompiledFromBuffers(
          embedderBytes: gestureEmbedderBytes,
          classifierBytes: gestureClassifierBytes,
          accelerators: accelerators,
          precision: precision,
        );
      } else {
        await _gestureRecognizer!.initializeFromBuffers(
          embedderBytes: gestureEmbedderBytes,
          classifierBytes: gestureClassifierBytes,
          performanceConfig: performanceConfig,
        );
      }
    }
  }

  /// Runs hand detection directly on the calling thread.
  Future<List<Hand>> detectDirect(cv.Mat image) async {
    if (_palm == null || _lm == null) {
      throw StateError(
        'HandDetectorCore not initialized. Call initializeFromBuffers() first.',
      );
    }

    final int imgW = image.cols;
    final int imgH = image.rows;

    // MediaPipe detection + tracking: reuse each tracked hand's ROI from the
    // previous frame, and only run the palm detector to discover new hands (or
    // whenever we are tracking fewer than [_maxDetections]). With tracking off
    // this reduces to running the palm detector every frame (original
    // behaviour).
    final List<PalmDetection> tracked =
        _enableTracking ? _trackedRois : const [];
    // MediaPipe gating (NormalizedRectVectorHasMinSizeCalculator): only run the
    // palm detector while we are tracking fewer than [_maxDetections] hands.
    final bool runPalm = !_enableTracking || tracked.length < _maxDetections;
    final List<PalmDetection> palms =
        runPalm ? await _palm!.detectOnMat(image) : const [];

    // MediaPipe AssociationNormRectCalculator: merge fresh palm-derived ROIs
    // (low priority) with the previous frame's landmark ROIs (high priority),
    // dropping either-side overlaps above [_trackAssocIou] so a re-detected
    // hand keeps its stable tracked ROI and duplicates are removed within each
    // list too. With tracking off this reduces to the per-frame palm list.
    List<PalmDetection> rois = _enableTracking
        ? associateRois(palms, tracked,
            imageWidth: imgW,
            imageHeight: imgH,
            minSimilarityThreshold: _tracking.associationIou)
        : palms;
    if (rois.length > _maxDetections) {
      // MediaPipe caps palm detections (ClipDetectionVectorSizeCalculator) and
      // leans on gating to bound the total; we cap defensively here. With
      // tracking on, associateRois appends the (gating-bounded) tracked ROIs
      // last, so keep the tail and a held hand never blinks out in favour of a
      // fresh palm. With tracking off, palms arrive in detector (score) order,
      // so keep the leading slots as the pre-tracking pipeline did.
      rois = _enableTracking
          ? rois.sublist(rois.length - _maxDetections)
          : rois.sublist(0, _maxDetections);
    }

    if (_mode == HandMode.boxes) {
      return _palmsToHands(image, rois);
    }

    final cropDataList = <_HandCropData>[];
    for (final roi in rois) {
      // Fused crop: warp the rotated square straight to the landmark model
      // input size in a single warpAffine (no native-size intermediate, no
      // separate resize). _buildResults scales landmarks back accordingly.
      final cropped = ImageUtils.rotateAndCropRectangle(
        image,
        roi,
        outSize: HandLandmarkModelRunner.inputSize,
      );
      if (cropped == null) continue;

      final (:cx, :cy, :size) = ImageUtils.palmCoordinates(roi, imgW, imgH);

      cropDataList.add(_HandCropData(
        palm: roi,
        croppedHand: cropped,
        rotation: roi.rotation,
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

    // _buildResults keeps hands whose landmark score clears the threshold and,
    // when tracking is on, records each kept hand's landmark-derived ROI for the
    // next frame. Hands that fall below the threshold are simply not carried
    // forward, so tracking drops them and the palm detector re-acquires later.
    _nextTrackedRois = _enableTracking ? <PalmDetection>[] : const [];
    final results = await _buildResults(image, cropDataList, allLandmarks);
    if (_enableTracking) _trackedRois = _nextTrackedRois;

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
      // The crop is resampled to the model input size, so landmark pixels are
      // in crop-input space; scale each back to original-image distance (this
      // is exactly the inverse of the warpAffine scale folded into the crop).
      final scaleX = data.cropSize / cropW;
      final scaleY = data.cropSize / cropH;
      final cosR = math.cos(data.rotation);
      final sinR = math.sin(data.rotation);

      for (final lm in lms.landmarks) {
        final xRel = (lm.x - cropW / 2) * scaleX;
        final yRel = (lm.y - cropH / 2) * scaleY;
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

      if (_enableTracking) {
        final nextRoi = _rectFromLandmarks(
            transformedLandmarks, image.cols, image.rows, lms.score);
        if (nextRoi != null) _nextTrackedRois.add(nextRoi);
      }
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

  /// Clears cross-frame tracking state. Call between unrelated inputs (a new
  /// video, or independent still images) so a stale ROI is not reused.
  void resetTracking() {
    _trackedRois = const [];
  }

  /// Builds the next-frame rotated ROI for a hand from its landmarks via the
  /// shared [roiFromHandLandmarks] (MediaPipe's landmarks-to-rect step).
  PalmDetection? _rectFromLandmarks(
      List<HandLandmark> lms, int imgW, int imgH, double score) {
    return roiFromHandLandmarks(
      xs: [for (final l in lms) l.x],
      ys: [for (final l in lms) l.y],
      imageWidth: imgW,
      imageHeight: imgH,
      score: score,
      scale: _tracking.roiScale,
      shiftY: _tracking.roiShiftY,
      minSqnSize: _tracking.minRoiSize,
      maxSqnSize: _tracking.maxRoiSize,
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
