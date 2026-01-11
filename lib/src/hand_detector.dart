import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'types.dart';
import 'image_utils.dart';
import 'palm_detector.dart';
import 'hand_landmark_model.dart';
import 'gesture_recognizer.dart';

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
/// This is a port of the Python hand detection library using the same models
/// and algorithms for anchor generation, box decoding, and rotation handling.
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
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances (1-10). Default: 1
  /// - [performanceConfig]: TensorFlow Lite performance configuration. Default: no acceleration
  /// - [enableGestures]: Whether to run gesture recognition. Default: false
  /// - [gestureMinConfidence]: Minimum confidence for gesture recognition (0.0-1.0). Default: 0.5
  HandDetector({
    this.mode = HandMode.boxesAndLandmarks,
    this.landmarkModel = HandLandmarkModel.full,
    this.detectorConf = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    this.performanceConfig = PerformanceConfig.disabled,
    this.enableGestures = false,
    this.gestureMinConfidence = 0.5,
  }) : interpreterPoolSize = performanceConfig.mode == PerformanceMode.disabled
            ? interpreterPoolSize
            : 1 {
    _palm = PalmDetector(scoreThreshold: detectorConf);
    _lm = HandLandmarkModelRunner(poolSize: this.interpreterPoolSize);
    if (enableGestures) {
      _gestureRecognizer = GestureRecognizer(minConfidence: gestureMinConfidence);
    }
  }

  /// Initializes the hand detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectOnImage].
  /// If already initialized, will dispose existing models and reinitialize.
  Future<void> initialize() async {
    if (_isInitialized) {
      await dispose();
    }

    // On desktop platforms the TensorFlow Lite C library must be loaded
    // into the process before creating any interpreters. This ensures the
    // native dylib/so is available for both palm and landmark models.
    await HandLandmarkModelRunner.ensureTFLiteLoaded();

    await _palm.initialize(performanceConfig: performanceConfig);
    await _lm.initialize(landmarkModel, performanceConfig: performanceConfig);

    // Initialize gesture recognizer if enabled
    if (_gestureRecognizer != null) {
      await _gestureRecognizer!.initialize(performanceConfig: performanceConfig);
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

    // Stage 1: Detect palms
    // NMS returns detections in descending score order (sorted before suppression)
    final List<PalmDetection> palms = await _palm.detectOnMat(image);

    // Debug assertion: verify NMS maintains score-descending order invariant
    assert(() {
      for (int i = 1; i < palms.length; i++) {
        if (palms[i].score > palms[i - 1].score) {
          throw StateError(
              'NMS output not sorted by score: ${palms[i - 1].score} < ${palms[i].score}');
        }
      }
      return true;
    }());

    // Limit detections (NMS guarantees highest-scored first)
    final limitedPalms = palms.length > maxDetections
        ? palms.sublist(0, maxDetections)
        : palms;

    if (mode == HandMode.boxes) {
      return _palmsToHands(image, limitedPalms, []);
    }

    // Stage 2: Crop, rotate, and extract landmarks

    // Phase 1: Preprocess all detections (crop and rotate)
    final cropDataList = <_HandCropData>[];
    for (final palm in limitedPalms) {
      final cropped = ImageUtils.rotateAndCropRectangle(image, palm);
      if (cropped == null) {
        continue;
      }

      // Calculate pixel coordinates for later transformation
      final centerX = palm.sqnRrCenterX * image.cols;
      final centerY = palm.sqnRrCenterY * image.rows;
      final size = palm.sqnRrSize * math.max(image.cols, image.rows);

      cropDataList.add(_HandCropData(
        palm: palm,
        croppedHand: cropped,
        rotation: palm.rotation,
        centerX: centerX,
        centerY: centerY,
        cropSize: size,
      ));
    }

    // Phase 2: Run landmark extraction in parallel for all hands
    final futures = cropDataList.map((data) async {
      try {
        return await _lm.run(data.croppedHand);
      } catch (_) {
        return null;
      }
    }).toList();

    final allLandmarks = await Future.wait(futures);

    // Phase 3: Post-process results and transform coordinates
    final results = await _buildResults(image, cropDataList, allLandmarks);

    // Clean up crop data (dispose cv.Mat objects)
    for (final data in cropDataList) {
      data.dispose();
    }

    return results;
  }

  /// Converts palm detections to Hand objects (boxes only mode).
  List<Hand> _palmsToHands(
    cv.Mat image,
    List<PalmDetection> palms,
    List<HandLandmarks?> landmarks,
  ) {
    final results = <Hand>[];

    for (int i = 0; i < palms.length; i++) {
      final palm = palms[i];

      // Calculate bounding box from rotation rectangle
      final centerX = palm.sqnRrCenterX * image.cols;
      final centerY = palm.sqnRrCenterY * image.rows;
      final size = palm.sqnRrSize * math.max(image.cols, image.rows);
      final halfSize = size / 2;

      results.add(Hand(
        boundingBox: BoundingBox(
          left: (centerX - halfSize).clamp(0, image.cols.toDouble()),
          top: (centerY - halfSize).clamp(0, image.rows.toDouble()),
          right: (centerX + halfSize).clamp(0, image.cols.toDouble()),
          bottom: (centerY + halfSize).clamp(0, image.rows.toDouble()),
        ),
        score: palm.score,
        landmarks: const [],
        imageWidth: image.cols,
        imageHeight: image.rows,
        handedness: i < landmarks.length ? landmarks[i]?.handedness : null,
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

      // Skip if landmark extraction failed or score too low
      if (lms == null || lms.score < minLandmarkScore) continue;

      // Transform landmarks from crop pixel space to original image space
      final transformedLandmarks = <HandLandmark>[];
      final cropW = data.croppedHand.cols.toDouble();
      final cropH = data.croppedHand.rows.toDouble();

      // Precompute trig values once per hand (used for all 21 landmarks)
      final cosR = math.cos(data.rotation);
      final sinR = math.sin(data.rotation);

      for (final lm in lms.landmarks) {
        // Landmarks are already in crop pixel space (after unpadding/rescaling)
        // No need to multiply by cropW/cropH - they're already in pixels
        final xCrop = lm.x;
        final yCrop = lm.y;

        // Apply inverse rotation to get original image coordinates
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

      // Run gesture recognition if enabled
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

      // Calculate bounding box from rotation rectangle
      final halfSize = data.cropSize / 2;

      results.add(Hand(
        boundingBox: BoundingBox(
          left: (data.centerX - halfSize).clamp(0, image.cols.toDouble()),
          top: (data.centerY - halfSize).clamp(0, image.rows.toDouble()),
          right: (data.centerX + halfSize).clamp(0, image.cols.toDouble()),
          bottom: (data.centerY + halfSize).clamp(0, image.rows.toDouble()),
        ),
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
    // Convert to relative position from crop center
    final xRel = xCrop - cropW / 2;
    final yRel = yCrop - cropH / 2;

    // Apply inverse rotation R(-rotation) to undo the forward R(+rotation)
    // R(-θ) = [cos(-θ), sin(-θ); -sin(-θ), cos(-θ)]
    //       = [cos(θ), -sin(θ); sin(θ), cos(θ)]
    final xRot = xRel * cosR - yRel * sinR;
    final yRot = xRel * sinR + yRel * cosR;

    // Translate to original image coordinates
    final xOrig = xRot + centerX;
    final yOrig = yRot + centerY;

    return (xOrig, yOrig);
  }
}
