import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'types.dart';
import 'util/image_utils.dart';
import 'models/palm_detector.dart';
import 'models/hand_landmark_model.dart';
import 'models/gesture_recognizer.dart';

/// Helper class to store preprocessing data for each detected palm.
///
/// Contains the palm detection info, preprocessed image, and transformation parameters
/// needed to convert landmark coordinates back to original image space.
class _HandCropData {
  /// The original palm detection result.
  final PalmDetection palm;

  /// The cropped and rotated hand image for landmark extraction.
  final cv.Mat croppedHand;

  /// Rotation angle in radians.
  final double rotation;

  /// Center X in original image pixels.
  final double centerX;

  /// Center Y in original image pixels.
  final double centerY;

  /// Size of the crop region in original image pixels.
  final double cropSize;

  _HandCropData({
    required this.palm,
    required this.croppedHand,
    required this.rotation,
    required this.centerX,
    required this.centerY,
    required this.cropSize,
  });

  /// Disposes the cv.Mat to free native memory.
  void dispose() {
    croppedHand.dispose();
  }
}

/// On-device hand detection and landmark estimation using TensorFlow Lite.
///
/// Implements a two-stage pipeline based on MediaPipe:
/// 1. Palm detection using SSD-based detector with rotation rectangle output
/// 2. Hand landmark model to extract 21 keypoints per detected hand
///
/// Uses the same models and algorithms as MediaPipe for anchor generation,
/// box decoding, and rotation handling.
///
/// Usage:
/// ```dart
/// final detector = HandDetector(
///   mode: HandMode.boxesAndLandmarks,
///   landmarkModel: HandLandmarkModel.full,
/// );
/// await detector.initialize();
/// final hands = await detector.detect(imageBytes);
/// await detector.dispose();
/// ```
class HandDetector {
  late final PalmDetector _palm;
  late final HandLandmarkModelRunner _lm;
  GestureRecognizer? _gestureRecognizer;

  /// Detection mode controlling pipeline behavior.
  final HandMode mode;

  /// Hand landmark model variant to use for landmark extraction.
  final HandLandmarkModel landmarkModel;

  /// Confidence threshold for palm detection (0.0 to 1.0).
  final double detectorConf;

  /// Maximum number of hands to detect per image.
  final int maxDetections;

  /// Minimum confidence score for landmark predictions (0.0 to 1.0).
  final double minLandmarkScore;

  /// Number of TensorFlow Lite interpreter instances in the landmark model pool.
  final int interpreterPoolSize;

  /// Performance configuration for TensorFlow Lite inference.
  ///
  /// By default, auto mode selects the optimal delegate per platform:
  /// - iOS: Metal GPU delegate
  /// - Android/macOS/Linux/Windows: XNNPACK (2-5x SIMD acceleration)
  final PerformanceConfig performanceConfig;

  /// Whether to run gesture recognition on detected hands.
  /// When enabled, each detected hand will include a [GestureResult].
  final bool enableGestures;

  /// Minimum confidence threshold for gesture recognition (0.0 to 1.0).
  /// Gestures with confidence below this threshold will be reported as [GestureType.unknown].
  final double gestureMinConfidence;

  bool _isInitialized = false;

  /// Creates a hand detector with the specified configuration.
  ///
  /// Parameters:
  /// - [mode]: Detection mode (boxes only or boxes + landmarks). Default: [HandMode.boxesAndLandmarks]
  /// - [landmarkModel]: Hand landmark model variant. Default: [HandLandmarkModel.full]
  /// - [detectorConf]: Palm detection confidence threshold (0.0-1.0). Default: 0.45
  /// - [maxDetections]: Maximum number of hands to detect. Default: 10
  /// - [minLandmarkScore]: Minimum landmark confidence score (0.0-1.0). Default: 0.5
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances (1-10). Default: 1.
  ///   Forced to 1 when a performance delegate (XNNPACK/auto) is enabled.
  /// - [performanceConfig]: TensorFlow Lite performance configuration. Default: auto (optimal per platform)
  /// - [enableGestures]: Whether to run gesture recognition. Default: false
  /// - [gestureMinConfidence]: Minimum confidence for gesture recognition (0.0-1.0). Default: 0.5
  HandDetector({
    this.mode = HandMode.boxesAndLandmarks,
    this.landmarkModel = HandLandmarkModel.full,
    this.detectorConf = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    this.performanceConfig = const PerformanceConfig(),
    this.enableGestures = false,
    this.gestureMinConfidence = 0.5,
  }) : interpreterPoolSize = performanceConfig.mode == PerformanceMode.disabled
            ? interpreterPoolSize
            : 1 {
    _palm = PalmDetector(scoreThreshold: detectorConf);
    _lm = HandLandmarkModelRunner(poolSize: this.interpreterPoolSize);
    if (enableGestures) {
      _gestureRecognizer =
          GestureRecognizer(minConfidence: gestureMinConfidence);
    }
  }

  /// Initializes the hand detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectOnMat].
  /// If already initialized, will dispose existing models and reinitialize.
  Future<void> initialize() async {
    if (_isInitialized) await dispose();

    await _initializeWith(
      palmLoader: () => _palm.initialize(performanceConfig: performanceConfig),
      landmarkLoader: () =>
          _lm.initialize(performanceConfig: performanceConfig),
      gestureLoader: (_) =>
          _gestureRecognizer!.initialize(performanceConfig: performanceConfig),
    );
  }

  /// Initializes the hand detector from pre-loaded model bytes.
  ///
  /// Used by [HandDetectorIsolate] to initialize within a background isolate
  /// where Flutter asset loading is not available.
  ///
  /// Parameters:
  /// - [palmDetectionBytes]: Raw bytes of the palm detection TFLite model
  /// - [handLandmarkBytes]: Raw bytes of the hand landmark TFLite model
  /// - [gestureEmbedderBytes]: Raw bytes of the gesture embedder model (optional; required for gesture recognition)
  /// - [gestureClassifierBytes]: Raw bytes of the gesture classifier model (optional; required for gesture recognition)
  Future<void> initializeFromBuffers({
    required Uint8List palmDetectionBytes,
    required Uint8List handLandmarkBytes,
    Uint8List? gestureEmbedderBytes,
    Uint8List? gestureClassifierBytes,
  }) async {
    if (_isInitialized) await dispose();

    await _initializeWith(
      palmLoader: () => _palm.initializeFromBuffer(
        palmDetectionBytes,
        performanceConfig: performanceConfig,
      ),
      landmarkLoader: () => _lm.initializeFromBuffer(
        handLandmarkBytes,
        performanceConfig: performanceConfig,
      ),
      gestureLoader:
          gestureEmbedderBytes != null && gestureClassifierBytes != null
              ? (_) => _gestureRecognizer!.initializeFromBuffers(
                    embedderBytes: gestureEmbedderBytes,
                    classifierBytes: gestureClassifierBytes,
                    performanceConfig: performanceConfig,
                  )
              : null,
    );
  }

  Future<void> _initializeWith({
    required Future<void> Function() palmLoader,
    required Future<void> Function() landmarkLoader,
    Future<void> Function(GestureRecognizer)? gestureLoader,
  }) async {
    await palmLoader();
    await landmarkLoader();

    if (_gestureRecognizer != null && gestureLoader != null) {
      await gestureLoader(_gestureRecognizer!);
    }

    _isInitialized = true;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Releases all resources used by the detector.
  Future<void> dispose() async {
    await _palm.dispose();
    await _lm.dispose();
    if (_gestureRecognizer != null) {
      await _gestureRecognizer!.dispose();
    }
    _isInitialized = false;
  }

  /// Detects hands in an image from raw bytes.
  ///
  /// Decodes the image bytes using OpenCV and performs hand detection.
  ///
  /// Parameters:
  /// - [imageBytes]: Raw image data in a supported format (JPEG, PNG, etc.)
  ///
  /// Returns a list of [Hand] objects, one per detected hand.
  /// Returns an empty list if image decoding fails or no hands are detected.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Hand>> detect(List<int> imageBytes) async {
    if (!_isInitialized) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }
    try {
      final mat = cv.imdecode(Uint8List.fromList(imageBytes), cv.IMREAD_COLOR);
      if (mat.isEmpty) return <Hand>[];
      try {
        return await detectOnMat(mat);
      } finally {
        mat.dispose();
      }
    } catch (e) {
      // Intentionally broad: imdecode can throw on malformed bytes; treat any
      // decode/pipeline failure as "no hands detected" for production robustness.
      return <Hand>[];
    }
  }

  /// Detects hands in an OpenCV Mat image.
  ///
  /// Performs the two-stage detection pipeline:
  /// 1. Detects palms using SSD-based detector with rotation rectangles
  /// 2. Crops and rotates hand regions, then extracts 21 landmarks per hand
  ///
  /// Parameters:
  /// - [image]: An OpenCV Mat in BGR format
  ///
  /// Returns a list of [Hand] objects, one per detected hand.
  /// Each hand contains:
  /// - A bounding box in original image coordinates
  /// - A confidence score (0.0-1.0)
  /// - 21 landmarks (if [mode] is [HandMode.boxesAndLandmarks])
  /// - Handedness (left or right)
  ///
  /// Note: The caller is responsible for disposing the input Mat after use.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Hand>> detectOnMat(cv.Mat image) async {
    if (!_isInitialized) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }

    final List<PalmDetection> palms = await _palm.detectOnMat(image);

    assert(() {
      for (int i = 1; i < palms.length; i++) {
        if (palms[i].score > palms[i - 1].score) {
          throw StateError(
              'NMS output not sorted by score: ${palms[i - 1].score} < ${palms[i].score}');
        }
      }
      return true;
    }());

    final limitedPalms =
        palms.length > maxDetections ? palms.sublist(0, maxDetections) : palms;

    if (mode == HandMode.boxes) {
      return _palmsToHands(image, limitedPalms);
    }

    final cropDataList = <_HandCropData>[];
    for (final palm in limitedPalms) {
      final cropped = ImageUtils.rotateAndCropRectangle(image, palm);
      if (cropped == null) {
        continue;
      }

      final (:centerX, :centerY, :size) = _palmCoordinates(palm, image);

      cropDataList.add(_HandCropData(
        palm: palm,
        croppedHand: cropped,
        rotation: palm.rotation,
        centerX: centerX,
        centerY: centerY,
        cropSize: size,
      ));
    }

    final futures = cropDataList.map((data) async {
      try {
        return await _lm.run(data.croppedHand);
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

  /// Converts palm detections to Hand objects (boxes only mode).
  List<Hand> _palmsToHands(
    cv.Mat image,
    List<PalmDetection> palms,
  ) {
    final results = <Hand>[];

    for (final palm in palms) {
      final (:centerX, :centerY, :size) = _palmCoordinates(palm, image);
      final halfSize = size / 2;

      results.add(Hand(
        boundingBox: _clampedBoundingBox(
            centerX, centerY, halfSize, image.cols, image.rows),
        score: palm.score,
        landmarks: const [],
        imageWidth: image.cols,
        imageHeight: image.rows,
        handedness: null,
        rotation: palm.rotation,
        rotatedCenterX: centerX,
        rotatedCenterY: centerY,
        rotatedSize: size,
      ));
    }

    return results;
  }

  /// Builds final Hand results with transformed landmark coordinates.
  ///
  /// Landmarks from the model runner are already in crop pixel space
  /// (after unpadding/rescaling to match Python's postprocessing).
  /// This method applies rotation and translation to transform them
  /// to original image coordinates.
  ///
  /// If gesture recognition is enabled, also runs gesture classification
  /// on each detected hand.
  Future<List<Hand>> _buildResults(
    cv.Mat image,
    List<_HandCropData> cropDataList,
    List<HandLandmarks?> allLandmarks,
  ) async {
    final results = <Hand>[];

    for (int i = 0; i < cropDataList.length; i++) {
      final data = cropDataList[i];
      final lms = allLandmarks[i];

      if (lms == null || lms.score < minLandmarkScore) continue;

      final transformedLandmarks = <HandLandmark>[];
      final cropW = data.croppedHand.cols.toDouble();
      final cropH = data.croppedHand.rows.toDouble();

      final cosR = math.cos(data.rotation);
      final sinR = math.sin(data.rotation);

      for (final lm in lms.landmarks) {
        final xCrop = lm.x;
        final yCrop = lm.y;

        final (xOrig, yOrig) = _transformToOriginal(
          xCrop,
          yCrop,
          cropW,
          cropH,
          cosR,
          sinR,
          data.centerX,
          data.centerY,
        );

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

  /// Transforms coordinates from crop space to original image space.
  ///
  /// Applies inverse rotation and translation to convert landmark
  /// coordinates from the rotated crop back to the original image.
  ///
  /// The forward transform in rotateAndCropRectangle applies R(+rotation) to the image.
  /// To match Python (hand_landmark.py:357), the inverse applies R(-rotation) to undo it.
  ///
  /// Parameters [cosR] and [sinR] are precomputed cos/sin of the rotation angle
  /// to avoid redundant trig calls when transforming multiple landmarks.
  (double, double) _transformToOriginal(
    double xCrop,
    double yCrop,
    double cropW,
    double cropH,
    double cosR,
    double sinR,
    double centerX,
    double centerY,
  ) {
    final xRel = xCrop - cropW / 2;
    final yRel = yCrop - cropH / 2;

    final xRot = xRel * cosR - yRel * sinR;
    final yRot = xRel * sinR + yRel * cosR;

    final xOrig = xRot + centerX;
    final yOrig = yRot + centerY;

    return (xOrig, yOrig);
  }

  ({double centerX, double centerY, double size}) _palmCoordinates(
    PalmDetection palm,
    cv.Mat image,
  ) {
    final (:cx, :cy, :size) =
        ImageUtils.palmCoordinates(palm, image.cols, image.rows);
    return (centerX: cx, centerY: cy, size: size);
  }

  BoundingBox _clampedBoundingBox(double centerX, double centerY,
      double halfSize, int imgWidth, int imgHeight) {
    return BoundingBox.ltrb(
      (centerX - halfSize).clamp(0, imgWidth.toDouble()),
      (centerY - halfSize).clamp(0, imgHeight.toDouble()),
      (centerX + halfSize).clamp(0, imgWidth.toDouble()),
      (centerY + halfSize).clamp(0, imgHeight.toDouble()),
    );
  }
}
